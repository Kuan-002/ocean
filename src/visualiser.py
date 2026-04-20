import math
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from PIL import Image
from numpy import asarray
from src.consensus_game import ConsensusGame
from src.consensus_game_env import ConsensusGameEnv
from src.config import Config
from src.utils import load_latest_checkpoint, reconstruct_autoencoder

def play_game(env, consensus_game, slots, attn, labels):
    env.reset(slots, attn, labels, False)
    consensus_game.zero_grad(set_to_none=True)
    with torch.no_grad():
        consensus_game.forward_game_inference(env)
    return env.slot_selection, env.all_claims, env.get_number_of_claims()

def set_up_visualisation(env_path, consensus_game_checkpoint_dir, sa_checkpoint_dir):
    config = Config(env_path, None)
    consensus_game = ConsensusGame(config)
    checkpoint = load_latest_checkpoint(consensus_game_checkpoint_dir)
    consensus_game.load(checkpoint)
    consensus_game.eval()
    env = ConsensusGameEnv(config, False)

    sa_checkpoint = load_latest_checkpoint(sa_checkpoint_dir)
    sa, _ = reconstruct_autoencoder(sa_checkpoint, config)
    return env, consensus_game, sa

def load_image(path):
    image = Image.open(path)
    image = torch.tensor(asarray(image))
    return image

def visualise_multiple(env, consensus_game, images, labels, slots, attn, slot_imgs, masks, max_k=6, save_path=None):
    rows = max(consensus_game.n_players, 2) * len(images)
    cols = max_k
    _, axes = plt.subplots(rows, cols + 1, figsize=(5 * cols, 3 * rows))

    for r in range(rows):
        axes[r][0].axis('off')

    if rows == 1:
        axes = [axes]

    for i in range(len(images)):
        all_selections, claims, _ = play_game(env, consensus_game, slots[i], attn[i], labels[i])
        final_claims = env.final_claims[0].tolist()
        plot_game(axes, i, consensus_game, images, all_selections, claims, final_claims, slot_imgs, masks, labels, max_k)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()

    plt.close()

def visualise(env, consensus_game, image, label, slots, attn, slot_imgs, masks, save_path=None):
    # Generate selections through play.
    all_selections, claims, turns = play_game(env, consensus_game, slots, attn, label)

    # Initialize the matplotlib figure
    rows = max(consensus_game.n_players, 2)
    cols = math.ceil(turns / consensus_game.n_players)
    _, axes = plt.subplots(rows, cols + 1, figsize=(5 * cols, 6))
    
    if rows == 1:
        axes = [axes]

    for r in range(rows):
        axes[r][0].axis('off')

    final_claims = env.final_claims[0].tolist()
    plot_game(axes, 0, consensus_game, [image], all_selections, claims, final_claims, slot_imgs, masks, [label], cols)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()

    plt.close()

def plot_game(axes, i, consensus_game, images, all_selections, claims, final_claims, slot_imgs, masks, labels, cols):
    row = i * max(consensus_game.n_players, 2)
    
    # Plot original image with label.
    axes[row][0].imshow(images[i])
    axes[row][0].set_title(f'Original Image: {labels[i]}')

    for j in range(max(consensus_game.n_players, 2)):
        for col in range(cols):
            axes[row + j][col + 1].axis('off')

    # Plot slot selections.
    for player_id in range(consensus_game.n_players):
        for selection_id in range(cols):
            if selection_id >= len(all_selections[0][player_id]) or all_selections[0][player_id][selection_id] == -1:
                axes[row + player_id][selection_id + 1].axis('off')
                continue

            selection = all_selections[0][player_id][selection_id].item()

            slot_img = slot_imgs[i][selection]
            mask = masks[i][selection]

            slot_recon = (((slot_img * mask + (1 - mask)) * 127.5) + 127.5).cpu().permute(1, 2, 0).numpy().clip(0, 255).astype('uint8')
            axes[row + player_id][selection_id + 1].imshow(slot_recon)
            claim = int(claims[0][player_id][selection_id])
            axes[row + player_id][selection_id + 1].set_title(f'Player {player_id + 1}, Slot {selection}, Claim {claim}')

    if len(set(final_claims)) == 1:
        axes[row + 1][0].text(0.5, 0.5, f'Agreed Classification: {int(final_claims[0])}', 
                                ha='center', va='center', fontsize=14, wrap=True)
    else:
        axes[row + 1][0].text(0.5, 0.5, 'Disagreement', 
                                ha='center', va='center', fontsize=14, wrap=True)

def visualize_slot_attention(model, image, num_slots=7, device="cuda", save_path=None):
    """
    Visualizes Slot Attention: Original Image, Individual Slots, and Reconstructed Image.
    """
    model.eval()  # Set the model to evaluation mode
    image = image.to(device)
    
    with torch.no_grad():
        # Add batch dimension
        input_image = image.unsqueeze(0)

        # Forward pass through the model
        reconstructed, slots, masks, all_slots, attn = model(input_image)

        # Convert reconstructed image back to original range (0, 255)
        reconstructed_image = ((reconstructed[0].cpu() * 127.5) + 127.5).clamp(0, 255).byte()

    # Convert the input image back to (0, 255) range for visualization
    original_image = ((image.permute(1, 2, 0).cpu() * 127.5) + 127.5).clamp(0, 255).numpy().astype('uint8')
    
    # Initialize the matplotlib figure
    _, axes = plt.subplots(4, num_slots // 2 + 2, figsize=(3 * (num_slots + 1), 6))
    
    for i in range(4):
        axes[i, 0].axis('off')

    # Plot the original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    
    # Plot the individual slot images
    for i in range(num_slots):
        # Slot attention images are extracted and reshaped (assuming they are in RGB format)
        slot_image = slots[0, i]
        mask_image = masks[0, i]
        
        slot_recon = (((slot_image * mask_image + (1 - mask_image)) * 127.5) + 127.5).cpu().permute(1, 2, 0).numpy().clip(0, 255).astype('uint8')

        mask_image = torch.ones_like(mask_image)
        slot_no_mask = (((slot_image * mask_image + (1 - mask_image)) * 127.5) + 127.5).cpu().permute(1, 2, 0).numpy().clip(0, 255).astype('uint8')
        
        j = 0
        k = 0
        if i > num_slots // 2:
            j = 1
            k = num_slots // 2 + 1
        
        axes[j, i + 1 - k].imshow(slot_recon)
        axes[j, i + 1 - k].set_title(f'Slot {i + 1}')
        axes[j, i + 1 - k].axis('off')

        axes[j + 2, i + 1 - k].imshow(slot_no_mask)
        axes[j + 2, i + 1 - k].set_title(f'Slot {i + 1}')
        axes[j + 2, i + 1 - k].axis('off')
    
    # Plot the reconstructed image
    axes[1, 0].imshow(reconstructed_image.permute(1, 2, 0))
    axes[1, 0].set_title('Reconstructed Image')

    # Hide unused axes if there are fewer slots than the number of columns
    if num_slots % 2 == 1:
        axes[1, -1].axis('off')
        axes[3, -1].axis('off')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

player_colors = ["#67886B", "#423B55", "#804E62", '#F7CAC9', '#92A8D1', '#955251', '#B565A7']

def visualise_fancy(env, consensus_game, image, label, slots, attn, slot_imgs, masks, class_names, save_path=None):
    # Generate selections through play.
    all_selections, claims, turns = play_game(env, consensus_game, slots, attn, label)

    n_players = consensus_game.n_players
    n_turn_rows = math.ceil(turns / n_players)
    total_rows = n_turn_rows + 3
    cols = max(n_players , 2)

    fig = plt.figure(figsize=(3 * cols, 2.8 * total_rows))
    outer_gs = gridspec.GridSpec(total_rows, 1, height_ratios=[1, 0.2] + [2] * n_turn_rows + [1], hspace=0.1)

    # === Top Row ===
    top_row = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[0], wspace=0.2)
    ax_img = fig.add_subplot(top_row[0])
    ax_class = fig.add_subplot(top_row[1])

    ax_img.imshow(image)
    ax_img.set_title("Original Image")
    ax_img.axis('off')

    ax_class.text(0.5, 0.5, f'True Class: {label}\n{class_names[label]}',
                  ha='center', va='center', fontsize=14, wrap=True)
    ax_class.axis('off')

    # Dialogue Title Row 
    ax_title = fig.add_subplot(outer_gs[1])
    ax_title.text(0.5, 0.2, "Dialogue", ha='center', va='center', fontsize=15, weight='bold')
    ax_title.axis('off')

    # Separator Line
    f = 1 / total_rows * 1.22
    if n_turn_rows == 3:
        f *= 1.05
    fig.add_artist(Line2D([0.05, 0.95], [1 - f, 1 - f], color='black', linewidth=2, transform=fig.transFigure))

    # Main Rows: Dialogue/Turns
    for row_idx in range(n_turn_rows):
        gs_row = gridspec.GridSpecFromSubplotSpec(1, cols, subplot_spec=outer_gs[row_idx + 2], wspace=0.5)
        for col_idx in range(n_players):
            ax = fig.add_subplot(gs_row[0, col_idx])

            # Slight downward shift for col > 0
            if col_idx > 0:
                pos = ax.get_position()
                ax.set_position([pos.x0, pos.y0 - 0.02 * col_idx, pos.width, pos.height])

            if row_idx >= len(claims[0][col_idx]) or claims[0][col_idx][row_idx] == -1:
                ax.axis('off')
                continue

            # Get subplot position for alignment
            pos = ax.get_position()
            img_x, img_y, img_w, img_h = pos.x0, pos.y0, pos.width, pos.height

            # Define avatar position and size
            avatar_size = img_h * 0.25
            avatar_x = img_x - avatar_size - 0.01
            avatar_y = img_y + img_h / 2 - avatar_size / 2

            # Create small axes for the avatar
            avatar_ax = fig.add_axes([avatar_x, avatar_y, avatar_size, avatar_size])
            avatar_ax.set_aspect('equal')
            avatar_ax.add_patch(Circle((0.5, 0.5), 0.3, color=player_colors[col_idx % len(player_colors)]))
            avatar_ax.text(0.5, 0.5, f'P{col_idx + 1}', ha='center', va='center', color='white', fontsize=8, weight='bold')
            avatar_ax.axis('off')

            selection = all_selections[0][col_idx][row_idx].item()
            claim = int(claims[0][col_idx][row_idx])
            if selection != -1:
                slot_img = slot_imgs[0][selection]
                mask = masks[0][selection]

                slot_recon = (((slot_img * mask + (1 - mask)) * 127.5) + 127.5).cpu().permute(1, 2, 0).numpy().clip(0, 255).astype('uint8')
                ax.imshow(slot_recon)
                ax.set_title(f'Slot {selection}')
                ax.text(0.5, -0.1, f'Claim: {claim}', ha='center', va='top', fontsize=10, transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f'Slot: Abstain. Claim: {claim}', ha='center', va='top', fontsize=10)

            ax.axis('off')

    ax_bottom = fig.add_subplot(outer_gs[-1])
    ax_bottom.axis('off')

    # Set classification text
    final_claims = env.final_claims[0].tolist()

    if len(set(final_claims)) == 1:
        ax_bottom.text(0.5, 0.65, f'Agreed Classification: {int(final_claims[0])}', 
                                ha='center', va='center', fontsize=14, wrap=True)
    else:
        ax_bottom.text(0.5, 0.65, 'Disagreement', 
                                ha='center', va='center', fontsize=14, wrap=True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()

    plt.close()
