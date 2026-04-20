import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.modules import RNNClassifier, RNNPolicyNetwork, Baseline
from src.config import Config

class Player(nn.Module):
    def __init__(self, config: Config):
        super(Player, self).__init__()

        self.max_norm = config.max_norm
        self.device = config.device
        self.use_baseline = config.use_baseline

        self.classifier = RNNClassifier(config.slot_dim, config.rnn_input_size, config.rnn_output_size, config.classifier_hidden, len(config.labels), config.use_gpu)
        self.model = RNNPolicyNetwork(config.slot_dim, config.rnn_input_size, config.rnn_output_size, config.player_hidden_dim, config.player_action_dim, config.num_slots, config.use_gpu)
        self.baseline = Baseline(config.rnn_output_size, config.baseline_hidden_dim, config.use_gpu)

        self.optimiser = optim.Adam(self.parameters(), lr=config.player_lr, weight_decay=1e-5)

        if config.use_gpu:
            self.cuda()

    def choose_action(self, state: tuple):
        [curr_player, claim_needed, all_slots, curr_slots, last_selection, is_last_turn, pre_claim_needed] = state

        # First turn of the game, initialise both RNNs.
        if last_selection[0][curr_player] == -1:
            self._init_hidden_state(curr_slots)

        # Apply selections from other players.
        self._apply_rnn_others(curr_player, last_selection, all_slots)

        # Get pre-claim if needed
        pre_claim_action = None
        if pre_claim_needed:
            pre_claim_logits = self.classifier.forward_classifier()
            pre_claim_probs = torch.softmax(pre_claim_logits, dim=-1)
            pre_claim = pre_claim_logits.argmax(dim=-1)
            pre_claim_action = (pre_claim, pre_claim_probs)

        # Compute the action probabilities
        selection_probs = self.model.forward_policy()

        if torch.isnan(selection_probs).any():
            raise ValueError("Nan probs detected!")

        # Choose an action by sampling from the probability distribution
        dist = torch.distributions.Categorical(selection_probs)
        selection = dist.sample()
        log_prob = dist.log_prob(selection)
        entropy = dist.entropy()

        # Add selection to both hidden states.
        self._apply_rnn(all_slots, selection)

        # Handle the claim if required
        claim = None
        classifier_logits = None
        if claim_needed:
            if is_last_turn:
                classifier_logits = self.classifier.forward_classifier()
            else:
                with torch.no_grad():
                    classifier_logits = self.classifier.forward_classifier()
            classifier_probs = torch.softmax(classifier_logits, dim=-1)
            claim = classifier_logits.argmax(dim=-1)

        # Compute the baseline for the current player
        if self.use_baseline:
            baseline = self.baseline.forward(self.model.hidden_state)
        else:
            baseline = torch.zeros((curr_slots.shape[0]), device=self.device)

        action = (selection_probs, log_prob, entropy, selection, classifier_logits, classifier_probs, claim, baseline)

        return action, pre_claim_action
    
    def learning_step(self, reinforce_loss: torch.Tensor, ce_loss: torch.Tensor, retain_graph: bool):
        total_loss = reinforce_loss + ce_loss
        total_loss.backward(retain_graph=retain_graph)

        if self.max_norm != -1.0:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_norm)

        self.optimiser.step()

    def _apply_rnn(self, slots: torch.Tensor, selection: torch.Tensor):
        slot_selections = slots[torch.arange(slots.size(0), device=self.device), selection]
        
        # Apply selection to classifier's RNN.
        self.classifier.forward_rnn(slot_selections)

        # Apply selection to model's RNN
        self.model.forward_rnn(slot_selections, selection)
        
    def _apply_rnn_others(self, player: int, last_selection: torch.Tensor, slots: torch.Tensor):
        if torch.all(last_selection == -1):
            return

        selections = -torch.ones((last_selection.size(0), last_selection.size(1) - 1), dtype=torch.long, device=self.device)

        if player != 0:
            selections[:, :player] = last_selection[:, :player]
        if player != last_selection.size(1) - 1:
            selections[:, player:] = last_selection[:, player + 1:]

        for i in range(selections.size(1)):
            if torch.all(selections[:, i] != -1):
                self._apply_rnn(slots, selections[:, i])
        
    def _init_hidden_state(self, slots: torch.Tensor):
        batch_size = slots.size(0)
        
        # Initialise policy's RNN
        self.model.reset_hidden(batch_size)
        for slot in range(slots.size(1)):
            self.model.forward_rnn(slots[:, slot], torch.full((batch_size,), slot, dtype=torch.long, device=self.device))

        # Initialise classifier's RNN
        self.classifier.reset_hidden(batch_size)
