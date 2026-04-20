import torch
import torch.nn as nn
import numpy as np
import random
import os
from src.slot_autoencoder import SlotAutoencoder
from src.slot_autoencoder import SlotAutoencoder
from src.consensus_game import ConsensusGame
from src.consensus_game_env import ConsensusGameEnv
from src.partial_visibility.consensus_game_env import ConsensusGameEnv as PVConsensusGameEnv

def init_params(p):
    if isinstance(p, (nn.Linear, nn.Conv2d, nn.Parameter)):
        nn.init.xavier_uniform_(p.weight)
        p.bias.data.fill_(0)

def seed_all(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def load_latest_checkpoint(checkpoint_dir, top=0):
    """Load the latest checkpoint from the given directory."""
    checkpoints = [
        os.path.join(checkpoint_dir, ckpt)
        for ckpt in os.listdir(checkpoint_dir)
        if ckpt.endswith(".pt")
    ]
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in the directory.")
    
    if top == 0:
        checkpoint = max(checkpoints, key=os.path.getctime)
    else:
        checkpoints.sort(key=os.path.getctime)
        if len(checkpoints) < top:
            raise ValueError(f"Only {len(checkpoints)} checkpoints found, but top={top} requested.")
        checkpoint = checkpoints[-top - 1]
    
    return checkpoint

def reconstruct_autoencoder(checkpoint, config):
    """Reconstruct the model from a checkpoint."""
    model = SlotAutoencoder(config)
    best_loss = model.load(checkpoint)

    return model, best_loss

def extract_slots_sa(data, sa, device):
    (images, _, classes) = data
    images = images.to(device)
    classes = classes.to(device)
    with torch.no_grad():
        _, _, _, slots, attn = sa(images)
        return slots, attn, classes

def setup_slot_attention(config, args, top=0):
    # Set up slot attention encoder-decoder model. 
    if args.reload_sa_checkpoint_path != '':
        if args.latest:
            checkpoint = load_latest_checkpoint(args.reload_sa_checkpoint_path, top)
        else:
            checkpoint = args.reload_sa_checkpoint_path

        model, best_loss = reconstruct_autoencoder(checkpoint, config)
        
        print(f"Loaded checkpoint: {checkpoint} with best loss: {best_loss}")
    else:
        model = SlotAutoencoder(config)

        model.apply(init_params)

        best_loss = 1e6
        
    return model, best_loss

def setup_consensus_game(config, args, top=0):
    if args.experiment == '':
        consensus_game_env = ConsensusGameEnv(config, True)
        eval_consensus_game_env = ConsensusGameEnv(config, False)
    elif args.experiment == 'partial_visibility':
        consensus_game_env = PVConsensusGameEnv(config, True)
        eval_consensus_game_env = PVConsensusGameEnv(config, False)
        config.player_action_dim = config.num_slots // config.n_players
    else:
        raise ValueError(f"Unknown experiment type: {args.experiment}")

    consensus_game = ConsensusGame(config)

    if args.reload_cg_checkpoint_path != '':
        if args.latest:
            checkpoint = load_latest_checkpoint(args.reload_cg_checkpoint_path, top)
        else:
            checkpoint = args.reload_cg_checkpoint_path
        consensus_game.load(checkpoint)

    return consensus_game_env, consensus_game, eval_consensus_game_env
