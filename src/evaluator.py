import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import islice
from src.consensus_game import ConsensusGame
from src.consensus_game_env import ConsensusGameEnv
from src.config import Config
from src.visualiser import visualise_multiple, load_image, visualize_slot_attention
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
 
class Evaluator:
    def __init__(self, slot_autoencoder: nn.Module, game: ConsensusGameEnv, cg: ConsensusGame, dataloader: DataLoader, config: Config):
        self.game = game
        self.cg = cg
        self.sa = slot_autoencoder
        self.dataloader = dataloader
        
        self.episodes = config.eval_episodes
        self.chart_frequency = config.chart_frequency
        if config.out_subpath != None:
            self.reward_temp_path = config.reward_temp_path
            self.losses_temp_path = config.losses_temp_path
            self.results_temp_path = config.results_temp_path
            self.metrics_temp_path = config.metrics_temp_path
            self.visualisation_temp_path = config.visualisation_temp_path
            self.sa_visualisation_temp_path = config.sa_visualisation_temp_path
        self.checkpoint = config.checkpoint
        self.num_slots = config.num_slots
        self.slot_dim = config.slot_dim
        self.device = config.device

        self.game_lengths = []
        self.empty_slot_percentages = []
        self.results = []
        self.rewards = {}
        self.losses = {}
        
        self.evaluations_cg = 0
        self.evaluations_sa = 0

        self.tqdm_interval = config.tqdm_interval

        # Visualisation
        if config.out_subpath != None and self.visualisation_temp_path is not None:
            self.setup_visualisation()

    def evaluate_cg(self, reinforce_losses: torch.Tensor, ce_losses: torch.Tensor, rewards: torch.Tensor, extract_slots: callable):
        self.track_reward(rewards)
        for i in range(self.cg.n_players):
            self.track_loss(reinforce_losses[i], "Policy Loss", f"Agent {i + 1}")
            self.track_loss(ce_losses[i], "Classifier Loss", f"Agent {i + 1}")

        accuracy = self.evaluate_training_cg(extract_slots)

        self.evaluations_cg += 1
        if self.evaluations_cg % self.chart_frequency == 0:
            self.plot_rewards(self.reward_temp_path)
            self.plot_losses(self.losses_temp_path)
            self.plot_results(self.results_temp_path)
            self.plot_metrics(self.metrics_temp_path)
            self.plot_visualisation(self.visualisation_temp_path)

        return accuracy

    def evaluate_sa(self, train_loss: float):
        valid_loss = self.evaluate_training_sa()
        self.track_loss(train_loss, "Slot Autoencoder", "Train Loss")
        self.track_loss(valid_loss, "Slot Autoencoder", "Validation Loss")

        self.evaluations_sa += 1
        if self.evaluations_sa % self.chart_frequency == 0:
            if self.sa_visualisation_temp_path is not None:
                sample_image = next(iter(self.dataloader))[0][0]
                visualize_slot_attention(self.sa, sample_image, num_slots=self.num_slots, device=self.device, save_path=self.sa_visualisation_temp_path)
            self.plot_losses(self.losses_temp_path)
        
        return valid_loss

    def evaluate_training_sa(self):
        self.sa.eval()
        loader = tqdm(
            enumerate(self.dataloader),
            total=len(self.dataloader),
            mininterval=self.tqdm_interval,
        )
        total_loss, count = 0, 0

        for _, (x, _, _) in loader:
            x = x.to(self.device)
            self.sa.zero_grad(set_to_none=True)
            with torch.no_grad():
                recon_combined, _, _, _, _ = self.sa(x)
            loss = self.sa.loss(x, recon_combined)
            loss = loss.detach()
            count += x.shape[0]
            total_loss += loss * x.shape[0]
            loader.set_description(
                "=> val | recon_loss: {:.8f}".format(total_loss / count),
                refresh=False,
            )
        return total_loss / count

    def evaluate_training_cg(self, extract_slots: callable):
        self.cg.eval()

        total_game_lengths = 0
        reward_dict = {}
        total_empty_slot_percentage = 0

        loader = tqdm(
            islice(enumerate(self.dataloader), self.episodes),
            total=self.episodes,
            mininterval=self.tqdm_interval,
        )
        loader.set_description("=> eval", refresh=False)

        for _, data in loader:
            # Set slots and labels on env.
            slots, attn, labels = extract_slots(data, self.sa, self.device)
            self.game.reset(slots, attn, labels)

            # Forward pass on cg.
            self.cg.zero_grad(set_to_none=True)
            with torch.no_grad():
                self.cg.forward_game_inference(self.game)

            # Track rewards
            _, reward_type = self.game.reward(get_types=True)
            
            reward_type = reward_type[0]            
            if reward_type != '':
                if reward_type not in reward_dict:
                    reward_dict[reward_type] = 0 
                reward_dict[reward_type] += 1

            # Track BG Slots
            total_empty_slot_percentage += self.game.get_percentage_empty_slots()

            total_game_lengths += self.game.get_slots_selected()

        self.game_lengths.append(total_game_lengths / self.episodes)
        self.empty_slot_percentages.append(total_empty_slot_percentage / self.episodes)
        self.results.append(reward_dict)

        self.cg.train()

        accuracy = reward_dict['Correct Consensus'] / self.episodes if 'Correct Consensus' in reward_dict else 0

        return accuracy

    def track_reward(self, rewards: torch.Tensor):
        for p, r in enumerate(rewards.detach()):
            if p not in self.rewards:
                self.rewards[p] = []
            self.rewards[p].append(r)

    def track_loss(self, loss: torch.Tensor, plot_name: str, agent_name: str):
        loss = loss.item()
        if plot_name not in self.losses:
            self.losses[plot_name] = {}
        
        if agent_name not in self.losses[plot_name]:
            self.losses[plot_name][agent_name] = []
        
        self.losses[plot_name][agent_name].append(loss)

    def plot_losses(self, file_path: str):
        if file_path is None or not self.losses:
            return

        num_plots = len(self.losses)
        num_cols = max([len(plot_dict) for plot_dict in self.losses.values()])

        if num_plots == 0:
            return

        fig, axes = plt.subplots(num_plots, num_cols, figsize=(12, 4 * num_plots))

        if num_plots == 1:
            axes = [axes]
        
        if num_cols == 1:
            axes = [[ax] for ax in axes]

        for p, (plot_name, plot_dict) in enumerate(self.losses.items()):
            for q, (player_name, loss_values) in enumerate(plot_dict.items()):
                ax = axes[p][q]
                ax.plot(loss_values)
                ax.set_title(f"{plot_name}, {player_name}")
                ax.set_xlabel(f'Epochs x {self.checkpoint}')
                ax.set_ylabel('Loss')

        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    def plot_rewards(self, file_path: str):
        if file_path is None or not self.rewards:
            return
        
        # Accumulate rewards per timestep.
        num_plots = len(self.rewards)
        num_stages = len(self.rewards[0][0])

        _, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

        if num_plots == 1:
            axes = [axes]

        for p, rewards in self.rewards.items():
            ax = axes[p]

            rewards = torch.stack(rewards).cpu().numpy()
                
            for k in range(num_stages):
                ax.plot(rewards[:, k], label=f"Stage {k + 1}")

            ax.legend()
            ax.set_title(f"Agent {p + 1}")
            ax.set_xlabel(f'Epochs x {self.checkpoint}')
            ax.set_ylabel('Rewards')

        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    def plot_results(self, file_path: str):
        if file_path is None or not self.results:
            return

        percent_results = {
            'Correct Consensus': [],
            'Incorrect Consensus': [],
            'No Consensus': [],
        }

        for result in self.results:
            total = sum(result.values())
            for key in percent_results.keys():
                if key in result:
                    percent_results[key].append(result[key] / total)
                else:
                    percent_results[key].append(0)

        for key, value in percent_results.items():
            plt.plot(value, label=key)

        plt.title('Results')
        plt.xlabel(f'Epochs x {self.checkpoint}')
        plt.ylabel('Percent')
        plt.legend()
        plt.savefig(file_path)
        plt.close()

    def plot_metrics(self, file_path: str):
        if file_path is None or not self.results:
            return

        num_plots = 4
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

        # Track consensus and accuracy.
        consensus_rates = []
        accuracy_rates = []
        for result in self.results:
            total = sum(result.values())
            consensus_count = 0
            true_count = 0
            if 'Correct Consensus' in result:
                consensus_count += result['Correct Consensus']
                true_count += result['Correct Consensus']
            if 'Incorrect Consensus' in result:
                consensus_count += result['Incorrect Consensus']
            consensus_rates.append(consensus_count / total)
            accuracy_rates.append(true_count / total)

        # Plot consensus rate.
        ax = axes[0]
        ax.plot(consensus_rates)
        ax.set_title("Consensus Rates")
        ax.set_xlabel(f'Epochs x {self.checkpoint}')
        ax.set_ylabel('Percentage')

        # Plot accuracy.
        ax = axes[1]
        ax.plot(accuracy_rates)
        ax.set_title("Accuracy Rates")
        ax.set_xlabel(f'Epochs x {self.checkpoint}')
        ax.set_ylabel('Percentage')

        # Plot game length.
        ax = axes[2]
        ax.plot(self.game_lengths)
        ax.set_title('Game Lengths')
        ax.set_xlabel(f'Epochs x {self.checkpoint}')
        ax.set_ylabel('Average Game Length')

        # Plot empty slot percentages.
        ax = axes[3]
        ax.plot(self.empty_slot_percentages)
        ax.set_title('Empty Slot Percentages')
        ax.set_xlabel(f'Epochs x {self.checkpoint}')
        ax.set_ylabel('Average Percentage of Empty Slots')

        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    def setup_visualisation(self):
        self.visualise_images = []
        self.visualise_labels = []
        dataloader_iter = iter(self.dataloader)
        for _ in range(3):
            image, _, label = next(dataloader_iter)
            image = image.to(self.device)
            self.visualise_images.append(image)
            self.visualise_labels.append(label.squeeze(0))                

    def plot_visualisation(self, file_path: str):
        if file_path is None:
            return

        images = []
        visualise_slots = []
        visualise_slot_imgs = []
        visualise_masks = []
        visualise_attn = []

        for i in range(3):
            with torch.no_grad():
                _, slot_imgs, masks, slots, attn = self.sa(self.visualise_images[i])
                images.append((self.visualise_images[i].squeeze(0).permute(1, 2, 0).cpu() * 127.5 + 127.5).clamp(0, 255).numpy().astype('uint8'))
                visualise_slots.append(slots)
                visualise_slot_imgs.append(slot_imgs.squeeze(0))
                visualise_masks.append(masks.squeeze(0))    
                visualise_attn.append(attn)

        visualise_multiple(
            self.game,
            self.cg,
            images,
            self.visualise_labels,
            visualise_slots,
            visualise_attn,
            visualise_slot_imgs,
            visualise_masks,
            max_k=4,
            save_path=file_path
        )

    def evaluate_all(self):
        self.sa.eval()
        self.cg.eval()

        loader = tqdm(
            enumerate(self.dataloader),
            total=len(self.dataloader),
            mininterval=self.tqdm_interval,
            disable=True
        )

        # Slot Attention: recon_loss
        # Consensus Game: Consensus Rate, Accuracy, Precision, Recall, F1
        # Consensus Game: Average Game length, Average Number of BG Slots, Average Slot Uniqueness.
        total_loss = 0
        total_game_lengths = 0
        total_slot_uniqueness = 0
        total_empty_slot_percentage = 0
        labels = []
        predictions = []

        for _, (x, _, label) in loader:
            x = x.to(self.device)
            label = label.to(self.device)
            self.sa.zero_grad(set_to_none=True)
            with torch.no_grad():
                recon_combined, _, _, slots, attn = self.sa(x)
            loss = self.sa.loss(x, recon_combined)
            loss = loss.detach()
            total_loss += loss

            # Set slots and labels on env.
            self.game.reset(slots, attn, label)

            # Forward pass on cg.
            self.cg.zero_grad(set_to_none=True)
            with torch.no_grad():
                prediction = self.cg.forward(self.game)

            # Track labels and predictions
            labels.append(label.cpu().item())
            predictions.append(prediction.cpu().item())

            # Track BG Slots
            total_empty_slot_percentage += self.game.get_percentage_empty_slots()
            
            # Track Game Length
            total_game_lengths += self.game.get_slots_selected()

            # Track Slot Uniqueness
            total_slot_uniqueness += self.game.get_slot_uniqueness()

        sa_loss = total_loss.item() / len(self.dataloader)
        avg_empty_slot_percentage = total_empty_slot_percentage / len(self.dataloader) * 100
        avg_game_length = total_game_lengths / len(self.dataloader)
        avg_slot_uniqueness = total_slot_uniqueness / len(self.dataloader)

        # Calculate metrics
        correct_consensus = sum(1 for l, p in zip(labels, predictions) if l == p)
        valid_indices = [i for i, p in enumerate(predictions) if p != -1]
        consensus = len(valid_indices)
        total = len(labels)

        valid_labels = [labels[i] for i in valid_indices]
        valid_preds = [predictions[i] for i in valid_indices]

        consensus_rate = consensus / total * 100
        accuracy = correct_consensus / total * 100
        precision = precision_score(valid_labels, valid_preds, average='macro') * consensus_rate
        recall = recall_score(valid_labels, valid_preds, average='macro') * consensus_rate
        f1 = f1_score(valid_labels, valid_preds, average='macro') * consensus_rate

        # Print metrics
        print(f"Slot Autoencoder Loss           : {sa_loss:.6f}")
        print(f"Consensus Rate                  : {consensus_rate:.2f}")
        print(f"Accuracy                        : {accuracy:.2f}")
        print(f"Precision (given Consensus)     : {precision:.2f}")
        print(f"Recall (given Consensus)        : {recall:.2f}")
        print(f"F1 Score (given Consensus)      : {f1:.2f}")
        print(f"Average Game Length             : {avg_game_length:.3f}")
        print(f"Average Empty Slot Percentage   : {avg_empty_slot_percentage:.2f}")
        print(f"Average Slot Uniqueness         : {avg_slot_uniqueness:.4f}")
        print(f"Total Evaluations               : {len(self.dataloader)}")

        return {
            "sa_loss": sa_loss,
            "consensus_rate": consensus_rate,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_game_length": avg_game_length,
            "avg_empty_slot_percentage": avg_empty_slot_percentage,
            "avg_slot_uniqueness": avg_slot_uniqueness,
            "preds": predictions,
            "labels": labels,
        }
    
    def generate_confusion(self, predictions, labels, save_path=None):
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(10, 8))
        label_set = sorted(set(labels + predictions))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_set, yticklabels=label_set)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
