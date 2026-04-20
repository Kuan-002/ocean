import argparse
import os
import sys

from src.config import Config
from src.clevr_hans_dataset import setup_dataloaders
from src.mds_dataset import setup_dataloaders_mds
from src.utils import setup_consensus_game, setup_slot_attention 
from src.evaluator import Evaluator

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

def to_csv(all_metrics, path):
    with open(path, 'w') as csv_file:
        csv_file.write("Confidence,Accuracy,Precision,Recall,F1,Consensus Rate,Avg Game Length,Empty Slot %,Slot Uniqueness,SA Loss\n")
        for result in all_metrics:
            csv_file.write(
                f"{result['accuracy']},"
                f"{result['precision']},{result['recall']},{result['f1']},"
                f"{result['consensus_rate']},{result['avg_game_length']},{result['avg_empty_slot_percentage']},"
                f"{result['avg_slot_uniqueness']},{result['sa_loss']}\n"
            )

    print(f"Results saved to {csv_file_path}")

if __name__ == '__main__':
    # Get file paths from args.
    parser = argparse.ArgumentParser(description='Training Object-Centric Visual Debates.')
    parser.add_argument('--latest', action="store_true", help='Load in the latest checkpoint from the reload_sa_checkpoint_path directory.')
    parser.add_argument('--reload_path', type=str, help='Path to out folder to reload.', default='')
    parser.add_argument('--env_path', type=str, help='Path to the .env file.', default='')
    parser.add_argument('--reload_sa_checkpoint_path', type=str, help='Path to the checkpoint file to reload.', default='')
    parser.add_argument('--reload_cg_checkpoint_path', type=str, help='Path to the checkpoint file to reload.', default='')
    parser.add_argument('--top', type=int, help='Number of checkpoints to check.', default=1)
    parser.add_argument('--confidence', type=float, help='Confidence Threshold', default=-1)
    parser.add_argument('--out_path', type=str, help='Path to out folder.', default='')
    parser.add_argument('--experiment', type=str, help='Type of experiment.', default='')
    args = parser.parse_args()

    # Set up log file
    if args.out_path != '':
        os.makedirs(args.out_path, exist_ok=True)
        tag = args.out_path.split('/')[-1]
        log_file_path = os.path.join(args.out_path, f'eval_{tag}.log')
        log_file = open(log_file_path, 'w')
        csv_file_path = os.path.join(args.out_path, f'results_{tag}.csv')
        sys.stdout = Tee(sys.__stdout__, log_file)

    # Fix up args
    if args.reload_path != '':
        if args.env_path == '':
            args.env_path = os.path.join(args.reload_path, ".env")
        if args.reload_sa_checkpoint_path == '':
            args.reload_sa_checkpoint_path = os.path.join(args.reload_path, "checkpoints", "sa")
        if args.reload_cg_checkpoint_path == '':
            args.reload_cg_checkpoint_path = os.path.join(args.reload_path, "checkpoints", "cg")

    # Set up environment. 
    config = Config(args.env_path, None)

    if config.dataset == "ch":
        dataloaders = setup_dataloaders(config, eval=True)
    elif config.dataset == "mds":
        dataloaders = setup_dataloaders_mds(config, eval=True)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    
    # Display Hyperparameters
    for k, v in sorted(config.__dict__.items()):
        print(f"--{k}={v}")

    all_metrics = []
    config.confidence_threshold = config.confidence_threshold if args.confidence == -1 else args.confidence

    for i in range(args.top - 1, -1, -1):
        print(f"Evaluating checkpoint {i + 1} of {args.top}...")

        # Load in slot attention
        model, best_loss = setup_slot_attention(config, args, i)

        # Set up consensus game
        consensus_game_env, consensus_game, eval_consensus_game_env = setup_consensus_game(config, args, i)

        # Evaluator
        evaluator = Evaluator(model, eval_consensus_game_env, consensus_game, dataloaders['test'], config)
        metrics = evaluator.evaluate_all()
        metrics["checkpoint"] = i
        all_metrics.append(metrics)

    # Generate Confusion Matrix for latest checkpoint
    if args.out_path != '':
        confusion_file_path = os.path.join(args.out_path, f'confusion_{tag}.png')
        evaluator.generate_confusion(metrics['preds'], metrics['labels'], confusion_file_path)

    # Write to csv
    if args.out_path != '':
        to_csv(all_metrics, csv_file_path)

    # Close file
    if args.out_path != '':
        log_file.close()
        sys.stdout = sys.__stdout__
        print(f"Evaluation logs saved to {log_file_path}")
