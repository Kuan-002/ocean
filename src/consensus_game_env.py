import torch
import random
from src.config import Config

class ConsensusGameEnv:
    def __init__(self, config: Config, training: bool):
        # Game Constants
        self.training = training
        self.batch_size = config.dataset_batch_size if training else 1
        self.num_slots = config.num_slots
        self.num_labels = len(config.labels)
        self.n_players = config.n_players
        self.device = config.device
        self.game_length = config.consensus_game_length
        self.end_condition = config.end_condition
        self.confidence_threshold = config.confidence_threshold
        self.pre_claim_needed = config.end_condition in ['confident_consensus', 'consensus']

        # Sparse Rewards
        self.penalise_selections = config.penalise_selections
        self.penalise_selections_weight = config.penalise_selections_weight
        self.gamma = config.player_gamma

        # Dense Rewards
        self.relative_confidence_change = config.relative_confidence_change
        self.always_classify = config.always_classify
        self.second_selection_uniqueness_factor = config.second_selection_uniqueness_factor
        self.repeat_selection_reward_factor = config.repeat_selection_reward_factor
        self.sparse_reward_factor = config.sparse_reward_factor
        self.informative_repeat_scaling = config.informative_repeat_scaling
        self.per_agent_uniqueness = config.per_agent_uniqueness

        # Game Variables
        self.curr_slots = None
        self.curr_attn = None
        self.curr_labels = None
        self.starting_player = 0
        self.curr_player = 0
        self.num_stages = 0

        self.slot_selection = -torch.ones((self.batch_size, self.n_players, self.game_length), dtype=torch.long, device=self.device)
        self.last_selection = -torch.ones((self.batch_size, self.n_players), dtype=torch.long, device=self.device)
        self.all_claims = -torch.ones((self.batch_size, self.n_players, self.game_length), dtype=torch.long, device=self.device)
        self.final_claims = -torch.ones((self.batch_size, self.n_players), dtype=torch.long, device=self.device)
        self.claims_probs = [[] for _ in range(self.n_players)]
        self.pre_claim_probs = [[] for _ in range(self.n_players)]

    def reset(self, slots: torch.Tensor, attn: torch.Tensor, labels: torch.Tensor, random_starting_player: bool = True):
        """Reset the game state and load new batch of data."""
        self.curr_slots = slots
        self.curr_attn = attn
        self.curr_labels = labels

        self.starting_player = random.randint(0, self.n_players - 1) if random_starting_player else 0
        self.curr_player = self.starting_player
        self.num_stages = 0
        self.slot_selection.fill_(-1)
        self.last_selection.fill_(-1)
        self.final_claims.fill_(-1)
        self.all_claims.fill_(-1)
        self.claims_probs = [[] for _ in range(self.n_players)]
        self.pre_claim_probs = [[] for _ in range(self.n_players)]

        return self.get_state()

    def get_state(self):
        """Return the current game state."""
        return [
            self.curr_player,       # shape: (1)
            self.is_claim_needed(), # shape: [B]
            self.curr_slots,        # shape: [B, num_slots, slot_size]
            self.curr_slots,        # shape: [B, num_slots, slot_size]
            self.last_selection,    # shape: [B, n_players]
            self.is_last_turn(),
            not self.training and self.pre_claim_needed,
        ]

    def pre_claim_step(self, action: tuple):
        """Perform the pre-claim step for the current player."""
        if not self.pre_claim_needed:
            raise RuntimeError("Pre-claim step is not supported during training or is not needed for this end condition.")

        pre_claim, claim_probs = action
        self.final_claims[:, self.curr_player] = pre_claim.clone().detach()
        self.pre_claim_probs[self.curr_player].append(claim_probs.clone().detach())
        self.all_claims[:, self.curr_player, self.num_stages] = pre_claim.clone().detach()

        done = self.is_game_over(True)

        return self.get_slots_selected() > 0 and done

    def step(self, actions: tuple):
        """Apply a batch of actions and update the state."""
        return self.do_action(self.curr_player, actions)

    def do_action(self, player: int, actions: tuple):
        """Apply a batch of actions for specific players."""
        slot_indices, claims, probs = actions  # slot_indices: [B], claims: [B] or None
        
        self.slot_selection[:, player, self.num_stages] = slot_indices.clone().detach()
        
        # We retain gradients for this one as this is used externally.
        self.last_selection[:, player] = slot_indices
        if claims is not None:
            self.final_claims[:, player] = claims.clone().detach()
            self.all_claims[:, player, self.num_stages] = claims.clone().detach()

        if probs is not None:
            self.claims_probs[player].append(probs.clone().detach())

        done = self.is_game_over()

        # Update player & stage
        self.curr_player = (self.curr_player + 1) % self.n_players
        if self.curr_player == self.starting_player:
            self.num_stages += 1

        return self.get_state(), done

    def is_claim_needed(self):
        """Return whether a claim is needed per batch."""
        return self.always_classify or (not self.training or self.num_stages == self.game_length - 1)

    def is_game_over(self, pre_claim_step: bool = False):
        """Return whether the game is over per batch."""
        max_reached = (((self.num_stages == self.game_length - 1) 
                        and (self.curr_player == (self.starting_player - 1) % self.n_players)) 
                        or (self.num_stages == self.game_length))

        if max_reached and not pre_claim_step:
            return True

        if self.training:
            return max_reached
        elif self.end_condition == "repeat":
            # When a player selects the same slot again.
            slot_selection = self.slot_selection[0]
            valid = slot_selection != -1

            has_duplicates =  any(
                torch.sum(valid[p]) > torch.unique(slot_selection[p][valid[p]]).numel()
                for p in range(self.n_players)
            )

            return has_duplicates
        elif self.end_condition == "consensus":
            # When all players agree on the same claim.
            claims = self.final_claims[0]
            consensus = torch.all(claims == claims[0])

            return consensus
        elif self.end_condition == "repeat_all":
            # When a player selects the same slot again.
            slot_selection = self.slot_selection[0]
            valid = slot_selection != -1

            all_selected = slot_selection[valid]
            has_duplicates = all_selected.numel() > torch.unique(all_selected).numel()

            return has_duplicates
        elif self.end_condition == "decreasing_confidence":
            probs = self.claims_probs[self.curr_player]
            if len(probs) < 2:
                return False
            
            curr_confidence = probs[-1][0].max()
            prev_confidence = probs[-2][0].max()

            return prev_confidence > curr_confidence
        elif self.end_condition == "confident_consensus":
            claims = self.final_claims[0]
            consensus = torch.all(claims == claims[0])
            probs = self.claims_probs[self.curr_player]
            pre_claim_probs = self.pre_claim_probs[self.curr_player]
            curr_confidence = probs[-1][0][claims[0]] if probs != [] else 0
            if pre_claim_probs != []:
                curr_confidence = max(curr_confidence, pre_claim_probs[-1][0][claims[0]])

            return consensus and curr_confidence >= self.confidence_threshold
        else:
            raise ValueError(f"Unknown end condition: {self.end_condition}")

    def reward_dense(self):
        probs = torch.stack([torch.stack(player_probs) for player_probs in self.claims_probs])
        probs = probs.permute(0, 2, 1, 3)

        _, B, K, _ = probs.shape
        all_rewards = torch.zeros((self.n_players, B, K), device=self.device)

        # Get confidence changes.
        labels = self.curr_labels.unsqueeze(1).expand(self.n_players, B, K)
        true_class_probs = torch.gather(probs, dim=3, index=labels.unsqueeze(-1)).squeeze(-1)

        if self.relative_confidence_change:
            rotation = torch.roll(torch.arange(self.n_players), shifts=-self.starting_player)
            true_class_probs = true_class_probs[rotation]
            interleaved = true_class_probs.permute(2, 1, 0).transpose(0, 1).reshape(B, -1)
            deltas = interleaved[:, 1:] - interleaved[:, :-1]
            deltas = torch.cat((interleaved[:, 0].unsqueeze(1), deltas), dim=1)
            rotation = torch.roll(torch.arange(self.n_players), shifts=self.starting_player)
            confidence_changes = deltas.reshape(B, K, -1).transpose(1, 2).permute(1, 0, 2)
            confidence_changes = confidence_changes[rotation]
            initial_rewards = confidence_changes
        else:
            initial_rewards = true_class_probs - 1 / self.num_labels

        # Provide rewards for two players.
        for i in range(self.n_players):
            p = (self.starting_player + i) % self.n_players

            # Reward confidence change.
            rewards = initial_rewards[p]

            # Get slot usefulness.
            slot_usefulness = torch.zeros((B, self.num_slots), device=self.device)
            batch_indices = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, self.game_length)
            slot_usefulness[batch_indices, self.slot_selection[:, p]] = rewards

            # Reward non-uniqueness of next selections.
            for j in range(1, K):
                # Reward if repeat
                current = self.slot_selection[:, p, j].unsqueeze(1)
                prev = self.slot_selection[:, p, :j]
                repeated = (current == prev).any(dim=1).float()

                # Reward repeats only after second selection.
                if j != 1:
                    rewards[:, j] += repeated * self.repeat_selection_reward_factor

                # Reward if repeat informative slot.
                usefulness = slot_usefulness[torch.arange(B, device=self.device), current.squeeze(-1)]
                rewards[:, j] += repeated * usefulness * self.informative_repeat_scaling

            # Reward uniqueness of second selection.
            if self.per_agent_uniqueness:
                rewards[:, 1] += (self.slot_selection[:, p, 1] != self.slot_selection[:, p, 0]) * self.second_selection_uniqueness_factor
            else:
                second_selections = self.slot_selection[:, :, 1]
                other_mask = (second_selections != second_selections[:, p].unsqueeze(1))
                other_mask[:, p] = True
                rewards[:, 1] += other_mask.all(dim=1)  * self.second_selection_uniqueness_factor
                
            all_rewards[p] = rewards

        return all_rewards

    def reward(self, get_types=False):
        B, _ = self.final_claims.shape

        rewards = torch.zeros(B, dtype=torch.float, device=self.device)

        # Mask 1: Any player didn't make a claim
        has_missing = (self.final_claims == -1).any(dim=1)
        
        # Return early.
        if has_missing.all():
            return rewards, None

        # Mask 2: All players agreed and are correct
        all_agree = (self.final_claims == self.final_claims[:, 0:1]).all(dim=1)
        correct_claim = (self.final_claims[:, 0] == self.curr_labels)
        is_correct_consensus = all_agree & correct_claim

        # Mask 3: All players agreed but are incorrect
        is_incorrect_consensus = all_agree & ~correct_claim

        # Mask 4: Disagreement (default case)
        is_disagreement = ~has_missing & ~all_agree

        # Create reward types
        reward_types = None
        if get_types:
            reward_types = [''] * B
            for i in range(B):
                if is_correct_consensus[i]:
                    reward_types[i] = 'Correct Consensus'
                elif is_incorrect_consensus[i]:
                    reward_types[i] = 'Incorrect Consensus'
                else:
                    reward_types[i] = 'No Consensus'

            return rewards, reward_types

        # Get dense rewards
        dense_rewards = self.reward_dense()

        if self.sparse_reward_factor == 0:
            return dense_rewards, reward_types

        # Apply rewards
        rewards[is_correct_consensus] = 1
        rewards[is_incorrect_consensus] = 0
        rewards[is_disagreement] = -1
        rewards[has_missing] = 0

        # Optional penalty
        if self.penalise_selections:
            penalties = self.selection_penalty()
            rewards -= self.penalise_selections_weight * penalties

        # Expand rewards to fit game length
        rewards = rewards.unsqueeze(1).expand(-1, self.game_length)

        # Apply discount
        discounts = torch.tensor([self.gamma ** t for t in range(self.game_length - 1, -1, -1)], device=self.device)
        rewards = rewards * discounts

        # Provide for all players
        rewards = rewards.unsqueeze(0).expand(self.n_players, -1, -1)

        # Apply dense rewards
        rewards = rewards * self.sparse_reward_factor
        rewards += dense_rewards

        return rewards, reward_types

    def selection_penalty(self):
        B = self.slot_selection.size(0)
        penalties = torch.zeros(B, dtype=torch.int8, device=self.slot_selection.device)

        for i in range(B):
            row = self.slot_selection[i]
            valid = row != -1
            penalties[i] = torch.unique(row[valid]).numel()

        return penalties

    def get_current_player(self):
        return self.curr_player
    
    def get_slots_selected(self):
        return self.num_stages * self.n_players + (self.curr_player - self.starting_player) % self.n_players

    def is_last_turn(self):
        return self.num_stages == self.game_length - 1
    
    def get_percentage_empty_slots(self):
        def _empty_slot_heuristic(attn):
            max_attn = attn.max(dim=-1).values
            var_attn = attn.var(dim=-1)
            split = attn.shape[-1] // 2
            tb_ratio_dev = abs((attn[:split].mean(dim=-1) / attn[split:].mean(dim=-1)).squeeze(0) - 1)                               
            return max_attn < 0.95 and var_attn < 0.01 and tb_ratio_dev < 0.5
        
        mask = self.slot_selection != -1
        selected_indices = self.slot_selection[mask]
        batch_indices = mask.nonzero(as_tuple=False)[:, 0]
        selected_attns = self.curr_attn[batch_indices, selected_indices]
        num_empty = 0

        for i in range(selected_attns.shape[0]):
            if _empty_slot_heuristic(selected_attns[i]):
                num_empty += 1

        total_slots = selected_indices.shape[0]
        return num_empty / total_slots if total_slots > 0 else 0.0
    
    def get_slot_uniqueness(self):
        all_selections = self.slot_selection[self.slot_selection != -1]
        if all_selections.numel() == 0:
            return 0.0
        
        unique_selections = torch.unique(all_selections)
        return unique_selections.numel() / all_selections.numel()

    def get_number_of_claims(self):
        """Return the number of claims made by all players."""
        return (self.all_claims != -1).sum(dim=1).sum().item()
