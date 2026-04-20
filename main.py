import argparse

from src.config import Config
from src.clevr_hans_dataset import setup_dataloaders
from src.mds_dataset import setup_dataloaders_mds
from src.utils import seed_all, setup_consensus_game, setup_slot_attention, extract_slots_sa 
from src.trainer import Trainer
from src.evaluator import Evaluator

def train_e2e_em(config, args, dataloaders):
    # Set up slot attention
    model, best_loss = setup_slot_attention(config, args)

    # Display Hyperparameters
    print(f"{model}\n#params: {sum(p.numel() for p in model.parameters()):,}")
    for k, v in sorted(config.__dict__.items()):
        print(f"--{k}={v}")

    # Set up consensus game
    consensus_game_env, consensus_game, eval_consensus_game_env = setup_consensus_game(config, args)

    # Trainer and evaluator
    trainer = Trainer(model, consensus_game_env, consensus_game, config, sa_best_loss=best_loss)
    evaluator = Evaluator(model, eval_consensus_game_env, consensus_game, dataloaders['val'], config)

    # Train
    trainer.train_e2e_em(dataloaders['train'], evaluator)

def train_e2e(config, args, dataloaders):
    # Set up slot attention
    model, best_loss = setup_slot_attention(config, args)

    # Display Hyperparameters
    print(f"{model}\n#params: {sum(p.numel() for p in model.parameters()):,}")
    for k, v in sorted(config.__dict__.items()):
        print(f"--{k}={v}")

    # Set up consensus game
    consensus_game_env, consensus_game, eval_consensus_game_env = setup_consensus_game(config, args)

    # Trainer and evaluator
    trainer = Trainer(model, consensus_game_env, consensus_game, config, sa_best_loss=best_loss)
    evaluator = Evaluator(model, eval_consensus_game_env, consensus_game, dataloaders['val'], config)

    # Train
    trainer.train_e2e(dataloaders['train'], evaluator, extract_slots_sa)

def train_slot_attention(config, args, dataloaders):
    # Set up slot attention
    model, best_loss = setup_slot_attention(config, args)

    # Display Hyperparameters
    print(f"{model}\n#params: {sum(p.numel() for p in model.parameters()):,}")
    for k, v in sorted(config.__dict__.items()):
        print(f"--{k}={v}")

    # Training Slot Attention
    trainer = Trainer(model, None, None, config)
    evaluator = Evaluator(model, None, None, dataloaders['val'], config)
    trainer.train_slot_attention(dataloaders['train'], evaluator=evaluator)

def train_pipeline(config, args, dataloaders):
    # Set up slot attention
    model, best_loss = setup_slot_attention(config, args)

    # Display Hyperparameters
    print(f"{model}\n#params: {sum(p.numel() for p in model.parameters()):,}")
    for k, v in sorted(config.__dict__.items()):
        print(f"--{k}={v}")

    # Set up consensus game
    consensus_game_env, consensus_game, eval_consensus_game_env = setup_consensus_game(config, args)

    # Trainer
    trainer = Trainer(model, consensus_game_env, consensus_game, config)

    evaluator = Evaluator(model, eval_consensus_game_env, consensus_game, dataloaders['val'], config)

    if args.train_slot_attention:
        trainer.train_slot_attention(dataloaders['train'], best_loss=best_loss, evaluator=evaluator)

    trainer.train_consensus_game(dataloaders['train'], extract_slots_sa, evaluator)

if __name__ == '__main__':
    # Get file paths from args.
    parser = argparse.ArgumentParser(description='Training Object-Centric Visual Debates.')
    parser.add_argument('--reload_sa_checkpoint_path', type=str, help='Path to the checkpoint file to reload.', default='')
    parser.add_argument('--reload_cg_checkpoint_path', type=str, help='Path to the checkpoint file to reload.', default='')
    parser.add_argument('--latest', action="store_true", help='Load in the latest checkpoint from the reload_sa_checkpoint_path directory.')
    parser.add_argument('--train_slot_attention', action="store_true", help='Train slot attention.')
    parser.add_argument('--type', type=str, help='Type of training.', default='e2e_em')
    parser.add_argument('--experiment', type=str, help='Type of experiment.', default='')
    parser.add_argument('--env_path', type=str, help='Path to the .env file.', default='.env')
    parser.add_argument('--out_subpath', type=str, help='Subdirectory for output files.', default='./out/run/')
    args = parser.parse_args()

    # Set up environment. 
    config = Config(args.env_path, args.out_subpath)
    seed_all(config.seed, config.deterministic)

    if config.dataset == "ch":
        dataloaders = setup_dataloaders(config)
    elif config.dataset == "mds":
        dataloaders = setup_dataloaders_mds(config)

    if args.type == "only_sa":
        train_slot_attention(config, args, dataloaders)
    elif args.type == "pipeline":
        train_pipeline(config, args, dataloaders)
    elif args.type == "e2e":
        train_e2e(config, args, dataloaders)
    elif args.type == "e2e_em":
        train_e2e_em(config, args, dataloaders)
    else:
        raise Exception(f"Unknown training type: {args.type}")
