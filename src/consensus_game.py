import torch
import torch.nn as nn
import torch.nn.functional as F
from src.agent import Player
from src.config import Config
from src.consensus_game_env import ConsensusGameEnv

class ConsensusGame(nn.Module):
    def __init__(self, config: Config):
        super(ConsensusGame, self).__init__()
        
        # Configs
        self.n_players = config.n_players
        self.entropy_beta = config.entropy_beta
        self.baseline_beta = config.baseline_beta
        self.redundancy_lambda = config.redundancy_lambda
        self.consensus_lambda = config.consensus_lambda
        self.train_only_last_classification = config.train_only_last_classification 
        self.device = config.device
        
        # Components in the Game
        self.players = [Player(config) for _ in range(self.n_players)]
        
        if config.use_gpu:
            self.cuda()

    def forward(self, env: ConsensusGameEnv):
        self.forward_game_inference(env)
        claims = env.final_claims
        predictions, _ = torch.mode(claims, dim=1)
        majority_count = (claims.size(1) // 2) + 1
        mode_counts = (claims == predictions.unsqueeze(1)).sum(dim=1)
        predictions[mode_counts < majority_count] = -1
        return predictions  # (B,)

    def forward_player(self, player_id: int, state: tuple):
        player = self.players[player_id]
        action = player.choose_action(state)
        return action
    
    def forward_game_inference(self, env: ConsensusGameEnv):
        # Reset game.
        state = env.get_state()
        done = False
        
        # Play game.
        while not done:
            # Choose current agent and action.
            curr_player = env.curr_player
            
            action, pre_claim_action = self.forward_player(curr_player, state)
            _, _, _, selection, _, classifier_probs, claim, _ = action

            # Special case pre-claim for agreement-based end conditions.
            if pre_claim_action is not None:
                done = env.pre_claim_step(pre_claim_action)
                if done:
                    break

            # Make a move.
            state, done = env.step((selection, claim, classifier_probs))
    
    def forward_game_train(self, env: ConsensusGameEnv):
        # Set up trackers.
        probs = [[] for _ in range(self.n_players)]
        log_probs = [[] for _ in range(self.n_players)]
        entropies = [[] for _ in range(self.n_players)]
        actions = [[] for _ in range(self.n_players)]
        all_classifier_logits = [[] for _ in range(self.n_players)]
        all_classifier_probs = [[] for _ in range(self.n_players)]
        baselines = [[] for _ in range(self.n_players)]

        # Reset game.
        state = env.get_state()
        done = False
        turn = 0
        
        # Play game.
        while not done:
            # Choose current agent and action.
            curr_player = env.curr_player
            
            action, _ = self.forward_player(curr_player, state)
            prob, log_prob, entropy, selection, classifier_logits, classifier_probs, claim, baseline = action

            # Make a move.
            state, done = env.step((selection, claim, classifier_probs))

            # Add to trackers
            probs[curr_player].append(prob)
            actions[curr_player].append(selection)
            log_probs[curr_player].append(log_prob)
            entropies[curr_player].append(entropy)
            baselines[curr_player].append(baseline)
            turn += 1

            # Track Classifier Logits
            if classifier_logits is not None:
                all_classifier_logits[curr_player].append(classifier_logits)
                all_classifier_probs[curr_player].append(classifier_probs)

        # Reshape probs and actions
        probs = [torch.stack(inner).permute(1, 0, 2) for inner in probs]
        actions = [torch.stack(inner).permute(1, 0) for inner in actions]
        log_probs = [torch.stack(inner).permute(1, 0) for inner in log_probs]
        entropies = [torch.stack(inner).permute(1, 0) for inner in entropies]
        all_classifier_logits = [torch.stack(inner).permute(1, 0, 2) for inner in all_classifier_logits]
        all_classifier_probs = [torch.stack(inner).permute(1, 0, 2) for inner in all_classifier_probs]
        baselines = [torch.stack(inner).permute(1, 0) for inner in baselines]

        return probs, log_probs, entropies, actions, all_classifier_logits, all_classifier_probs, baselines

    def training_step(self, 
                      probs: list,
                      log_probs: list, 
                      entropies: list,
                      baselines: list, 
                      rewards: torch.tensor, 
                      logits: list, 
                      classifier_probs: list, 
                      labels: torch.Tensor, 
                      retain_graph: bool = False, 
                      reinforce_loss_weight: float = 1, 
                      ce_loss_weight: float = 1):
        reinforce_losses = [0 for _ in range(self.n_players)]
        ce_losses = [0 for _ in range(self.n_players)]

        # Calculate loss for each player
        for i in range(self.n_players):
            # Compute REINFORCE loss
            reinforce_loss = self._reinforce_loss(log_probs[i], rewards[i], entropies[i], baselines[i])
            redundancy_loss = self._redundancy_loss(probs[i])
            regularisation_loss = 0
            if self.redundancy_lambda != 0:
                regularisation_loss = self.redundancy_lambda * redundancy_loss
            reinforce_losses[i] = reinforce_loss + regularisation_loss

            # Compute cross-entropy loss
            ce_loss = self._ce_loss(logits[i], labels)
            consensus_loss = 0
            if self.n_players > 1 and self.consensus_lambda != 0:
                consensus_loss = self._consensus_loss(classifier_probs[i], torch.stack([classifier_probs[j] for j in range(self.n_players) if j != i]))
            ce_losses[i] = ce_loss + self.consensus_lambda * consensus_loss

            reinforce_losses[i] *= reinforce_loss_weight
            ce_losses[i] *= ce_loss_weight

            self.players[i].learning_step(reinforce_losses[i], ce_losses[i], retain_graph or (i != self.n_players - 1))

        return torch.stack(reinforce_losses).detach(), torch.stack(ce_losses).detach()
    
    def _reinforce_loss(self, log_probs: torch.Tensor, rewards: torch.Tensor, entropy: torch.Tensor, baselines: torch.Tensor):
        advantage = rewards - baselines.detach()

        # REINFORCE policy loss
        policy_loss = -torch.mean(log_probs * advantage)
        
        # Entropy term to encourage exploration
        entropy_loss = -torch.mean(entropy)

        # Baseline loss
        baseline_loss = F.mse_loss(baselines, rewards)

        total_loss = policy_loss + self.entropy_beta * entropy_loss + self.baseline_beta * baseline_loss  
        
        return total_loss

    def _redundancy_loss(self, probs: torch.tensor):
        mean_probs = probs.mean(dim=1)
        var = torch.var(mean_probs, dim=1, unbiased=False)
        return -var.mean()

    def _ce_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        if self.train_only_last_classification:
            logits = logits[:, -1:]

        B, K, C = logits.size()
        logits_flattened = logits.reshape(-1, C)
        labels_expanded = labels.unsqueeze(1).expand(B, K).reshape(-1)
        criterion = torch.nn.CrossEntropyLoss()
        ce_loss = criterion(logits_flattened, labels_expanded)
        return ce_loss

    def _consensus_loss(self, main_probs: torch.Tensor, target_probs: torch.Tensor):
        if self.train_only_last_classification:
            main_probs = main_probs[:, -1:]
            target_probs = target_probs[:, :, -1:]
        consensus_probs = target_probs.mean(dim=0)
        main_flat = main_probs.reshape(-1, main_probs.size(-1)).clamp(min=1e-8)
        consensus_flat = consensus_probs.reshape(-1, consensus_probs.size(-1)).clamp(min=1e-8)

        # Compute KL divergence
        loss = F.kl_div(main_flat.log(), consensus_flat.detach(), reduction='batchmean')
        return loss

    def save(self, path: int):
        to_save = {
            'agent_state_dicts': [agent.state_dict() for agent in self.players],
        }
        torch.save(to_save, path)

    def load(self, path: int):
        checkpoint = torch.load(path, weights_only=True)
        for i, agent in enumerate(self.players):
            agent.load_state_dict(checkpoint['agent_state_dicts'][i])

    def eval(self):
        super().eval()
        for player in self.players:
            player.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        for player in self.players:
            player.train(mode)
        return self
    
    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for agent in self.players:
            agent.zero_grad(set_to_none=set_to_none)
