import torch
from tqdm import tqdm
from src.slot_autoencoder import SlotAutoencoder
from src.consensus_game_env import ConsensusGameEnv
from src.consensus_game import ConsensusGame
from src.config import Config
from torch.utils.data import DataLoader
from src.evaluator import Evaluator
from src.utils import extract_slots_sa

class Trainer:
    def __init__(self, slot_autoencoder: SlotAutoencoder, consensus_game_env: ConsensusGameEnv, consensus_game: ConsensusGame, config: Config, sa_best_loss: float = None, cg_best_loss: float = None):
        self.sa = slot_autoencoder
        self.env = consensus_game_env
        self.cg = consensus_game

        self.batch_size = config.dataset_batch_size
        self.epochs = config.epochs
        self.checkpoint = config.checkpoint
        self.checkpoint_path_cg = config.checkpoint_path_cg
        self.checkpoint_path_sa = config.checkpoint_path_sa
        self.best_avg_rewards = config.reward_threshold
        self.device = config.device
        self.loss_threshold = config.loss_threshold
        self.tqdm_interval = config.tqdm_interval
        self.debug = config.debug
        self.save_freq = config.save_freq
        self.reinforce_loss_weight = config.reinforce_loss_weight
        self.ce_loss_weight = config.ce_loss_weight
        self.sa_loss_weight = config.sa_loss_weight
        self.em_reward_sa_loss_scaling = config.em_reward_sa_loss_scaling
        self.min_save_epochs = config.min_save_epochs

        if sa_best_loss != None:
            self.sa_best_loss = sa_best_loss
        else:
            self.sa_best_loss = self.loss_threshold

        if cg_best_loss != None:
            self.cg_best_loss = cg_best_loss
        else:
            self.cg_best_loss = self.loss_threshold

        self.best_accuracy = 0

    def train_e2e_em(self, dataloader: DataLoader, evaluator: Evaluator):
        print("Sanity Check")
        evaluator.evaluate_training_sa()
        evaluator.evaluate_training_cg(extract_slots_sa)
        print("\nTraining Slot Attention and Consensus Game with Expectation Maximisation")
        for epoch in range(self.epochs):
            print("\nEpoch {}:".format(epoch))

            # E-Step: Train CG (Freeze SA)
            cg_reinforce_losses, cg_ce_losses, cg_avg_rewards = self.run_epoch_consensus_game(dataloader, extract_slots_sa)

            # M-Step: Train SA (Freeze CG)
            sa_train_loss = self.run_epoch_consensus_game_train_sa(dataloader)

            if (epoch + 1) % self.checkpoint == 0:
                sa_valid_loss = evaluator.evaluate_sa(sa_train_loss)
                accuracy = evaluator.evaluate_cg(cg_reinforce_losses, cg_ce_losses, cg_avg_rewards, extract_slots_sa)
                self.save_checkpoints(epoch, sa_valid_loss, accuracy)

    def train_e2e(self, dataloader: DataLoader, evaluator: Evaluator, extract_slots: callable):
        print("Sanity Check")
        evaluator.evaluate_training_sa()
        evaluator.evaluate_training_cg(extract_slots)
        print("\nTraining Slot Attention and Consensus Game")
        for epoch in range(self.epochs):
            print("\nEpoch {}:".format(epoch))
            sa_train_loss, cg_reinforce_losses, cg_ce_losses, cg_avg_rewards = self.run_epoch_e2e(dataloader)

            if (epoch + 1) % self.checkpoint == 0:
                sa_valid_loss = evaluator.evaluate_sa(sa_train_loss)
                accuracy = evaluator.evaluate_cg(cg_reinforce_losses, cg_ce_losses, cg_avg_rewards, extract_slots)
                self.save_checkpoints(epoch, sa_valid_loss, accuracy)
            
    def train_slot_attention(self, dataloader: DataLoader, evaluator: Evaluator):
        print("Sanity Check")
        evaluator.evaluate_training_sa()
        print("\nTraining Slot Attention")
        for epoch in range(self.epochs):
            print("\nEpoch {}:".format(epoch))
            train_loss = self.run_epoch_slot_attention(dataloader)

            if (epoch + 1) % self.checkpoint == 0:
                valid_loss = evaluator.evaluate_sa(train_loss)
                self.save_checkpoint_sa(epoch, valid_loss)

    def train_consensus_game(self, dataloader: DataLoader, extract_slots: callable, evaluator: Evaluator):
        print("Sanity Check")
        evaluator.evaluate_training_cg(extract_slots)
        print("\nTraining Consensus Game")
        for epoch in range(self.epochs):
            print("\nEpoch {}:".format(epoch))
            # Run a single epoch of training
            reinforce_losses, ce_losses, avg_rewards = self.run_epoch_consensus_game(dataloader, extract_slots)
            
            # Save model if loss has improved.
            reference_loss = sum(ce_losses).item()

            # Evaluate and save during checkpoints.
            if (epoch + 1) % self.checkpoint == 0:
                accuracy = evaluator.evaluate_cg(reinforce_losses, ce_losses, avg_rewards, extract_slots)

                self.save_checkpoint_cg(epoch, reference_loss, avg_rewards.mean(), accuracy)

    def run_epoch_e2e(self, dataloader: DataLoader):
        self.sa.train()
        self.cg.train()
        loader = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            mininterval=self.tqdm_interval,
        )

        total_sa_loss = 0
        total_cg_reinforce_losses = 0
        total_cg_ce_losses = 0
        total_cg_rewards = 0

        for i, (x, _, labels) in loader:
            x = x.to(self.device)
            labels = labels.to(self.device)

            # Slot Attention
            self.sa.zero_grad(set_to_none=True)
            recon_combined, _, _, slots, attn = self.sa(x)
            sa_loss = self.sa.loss(x, recon_combined)

            # Consensus Game
            self.env.reset(slots, attn, labels)
            self.cg.zero_grad(set_to_none=True)
            probs, log_probs, entropies, _, classifier_logits, classifier_probs, baselines = self.cg.forward_game_train(self.env)
            rewards, _ = self.env.reward()
            reinforce_losses, ce_losses = (
                self.cg.training_step(
                    probs,
                    log_probs,
                    entropies,
                    baselines,
                    rewards,
                    classifier_logits,
                    classifier_probs,
                    self.env.curr_labels,
                    True,
                    self.reinforce_loss_weight,
                    self.ce_loss_weight
                )
            )

            # Apply weighting.
            sa_loss = self.sa_loss_weight * sa_loss

            # Backprop.
            sa_loss = self.sa.training_step(sa_loss)

            # Track losses.
            total_cg_reinforce_losses += reinforce_losses
            total_cg_ce_losses += ce_losses
            total_cg_rewards += rewards.mean(dim=1)

            total_sa_loss += sa_loss

            # Update loader.
            loader.set_description(
                "=> train | recon_loss: {:.8f} | ce_loss: {:.5f} | reinforce_loss: {:.5f}".format(
                    total_sa_loss / (i + 1),
                    ce_losses.sum() / self.cg.n_players,
                    reinforce_losses.sum() / self.cg.n_players
                ),
                refresh=False,
            )

        avg_sa_loss = total_sa_loss / len(dataloader)
        avg_reinforce_losses = total_cg_reinforce_losses / len(dataloader)
        avg_ce_losses = total_cg_ce_losses / len(dataloader)
        avg_cg_rewards = total_cg_rewards / len(dataloader)

        return avg_sa_loss, avg_reinforce_losses, avg_ce_losses, avg_cg_rewards

    def run_epoch_consensus_game_train_sa(self, dataloader: DataLoader):
        self.sa.train()
        loader = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            mininterval=self.tqdm_interval,
        )

        total_sa_loss = 0
        total_aug_loss = 0

        for i, (x, _, labels) in loader:
            x = x.to(self.device)
            labels = labels.to(self.device)

            # Slot Attention
            self.sa.zero_grad(set_to_none=True)
            recon_combined, _, _, slots, attn = self.sa(x)
            sa_loss = self.sa.loss(x, recon_combined)

            # Consensus Game
            self.env.reset(slots, attn, labels)
            self.cg.zero_grad(set_to_none=True)
            with torch.no_grad():
                self.cg.forward_game_train(self.env)
            rewards, _ = self.env.reward()

            # Apply weighting to only mean reward.
            aug_loss = sa_loss - self.em_reward_sa_loss_scaling * rewards.mean()

            # Backprop.
            aug_loss = self.sa.training_step(aug_loss)
            sa_loss = sa_loss.detach()

            # Track losses.
            total_sa_loss += sa_loss
            total_aug_loss += aug_loss

            # Update loader.
            loader.set_description(
                "=> train | recon_loss: {:.8f} | aug_loss: {:.8f}".format(
                    total_sa_loss / (i + 1),
                    total_aug_loss / (i + 1),
                ),
                refresh=False,
            )

        avg_sa_loss = total_sa_loss / len(dataloader)

        return avg_sa_loss

    def run_epoch_slot_attention(self, dataloader: DataLoader):
        self.sa.train()
        loader = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            mininterval=self.tqdm_interval,
        )
        total_loss, count = 0, 0

        for _, (x, _, _) in loader:
            x = x.to(self.device)
            self.sa.zero_grad(set_to_none=True)
            recon_combined, _, _, _, _ = self.sa(x)
            loss = self.sa.loss(x, recon_combined)
            self.sa.training_step(loss)
            loss = loss.detach()
            count += x.shape[0]
            total_loss += loss * x.shape[0]
            loader.set_description(
                "=> train | recon_loss: {:.8f}".format(total_loss / count),
                refresh=False,
            )

        return total_loss / count

    def run_epoch_consensus_game(self, dataloader: DataLoader, extract_slots: callable):
        self.cg.train()

        loader = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            mininterval=self.tqdm_interval,
        )

        total_reinforce_losses = 0
        total_ce_losses = 0
        total_rewards = 0

        for _, data in loader:
            # Set slots and labels on env.
            slots, attn, labels = extract_slots(data, self.sa, self.device)
            self.env.reset(slots, attn, labels)

            # Forward pass on cg.
            self.cg.zero_grad(set_to_none=True)
            probs, log_probs, entropies, _, classifier_logits, classifier_probs, baselines = self.cg.forward_game_train(self.env)
            
            # Get rewards
            rewards, _ = self.env.reward()

            # Perform training step.
            reinforce_losses, ce_losses = (
                self.cg.training_step(
                    probs,
                    log_probs,
                    entropies,
                    baselines,
                    rewards,
                    classifier_logits,
                    classifier_probs,
                    self.env.curr_labels,
                    reinforce_loss_weight=self.reinforce_loss_weight,
                    ce_loss_weight=self.ce_loss_weight,
                )
            )

            # Track losses.
            total_reinforce_losses += reinforce_losses
            total_ce_losses += ce_losses
            total_rewards += rewards.mean(dim=1)

            # Update tqdm loader
            loader.set_description(
                "=> train | ce_loss: {:.5f} | reinforce_loss: {:.5f}".format(ce_losses.sum() / self.cg.n_players, reinforce_losses.sum() / self.cg.n_players),
                refresh=False,
            )

        avg_reinforce_losses = reinforce_losses / len(dataloader)
        avg_ce_losses = ce_losses / len(dataloader)
        avg_rewards = total_rewards / len(dataloader)

        return avg_reinforce_losses, avg_ce_losses, avg_rewards

    def save_checkpoint_cg(self, epoch: int, reference_loss: torch.Tensor, avg_rewards: float, accuracy: float):
        save = False
        if reference_loss < self.cg_best_loss:
            self.cg_best_loss = reference_loss
            save= True

        if avg_rewards > self.best_avg_rewards:
            self.best_avg_rewards = avg_rewards
            save = True

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            save = True

        if (epoch + 1) % (self.checkpoint * self.save_freq) == 0 and epoch > self.min_save_epochs:
            save = True

        if not save:
            return

        self.cg.save(f"{self.checkpoint_path_cg}{epoch}_ckpt.pt")
        print(f"CG Model saved at epoch {epoch + 1} with loss {reference_loss:.5f}")

    def save_checkpoint_sa(self, epoch: int, reference_loss: torch.Tensor):
        save = False
        if reference_loss < self.sa_best_loss:
            save = True
            self.sa_best_loss = reference_loss

        if (epoch + 1) % (self.checkpoint * self.save_freq) == 0 and epoch > self.min_save_epochs:
            save = True
        
        if not save:
            return
        
        self.sa.save(f"{self.checkpoint_path_sa}{epoch}_ckpt.pt", reference_loss)
        print(f"SA Model saved at epoch {epoch + 1} with loss {reference_loss:.8f}")

    def save_checkpoints(self, epoch: int, sa_loss: float, accuracy: float):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.sa.save(f"{self.checkpoint_path_sa}{epoch}_ckpt.pt", sa_loss)
            self.cg.save(f"{self.checkpoint_path_cg}{epoch}_ckpt.pt")
            print(f"SA and CG saved at epoch {epoch + 1}. New best accuracy: {self.best_accuracy:.4f}.")
