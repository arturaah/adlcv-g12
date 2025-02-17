import torch
import matplotlib.pyplot as plt
import seaborn as sns
from vit import ViT, positional_encoding_2d
import numpy as np

def plot_positional_encodings(fixed_pos_enc, learned_pos_enc, save_path=None):
    """
    Visualizes and compares fixed and learned positional encodings
    Args:
        fixed_pos_enc: tensor of shape (num_patches, embed_dim)
        learned_pos_enc: tensor of shape (1, num_patches, embed_dim)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot fixed positional encoding
    sns.heatmap(fixed_pos_enc.numpy(), ax=axes[0,0], cmap='viridis')
    axes[0,0].set_title('Fixed Positional Encoding')
    axes[0,0].set_xlabel('Embedding Dimension')
    axes[0,0].set_ylabel('Position')
    
    # Plot learned positional encoding
    sns.heatmap(learned_pos_enc.squeeze(0).numpy(), ax=axes[0,1], cmap='viridis')
    axes[0,1].set_title('Learned Positional Encoding')
    axes[0,1].set_xlabel('Embedding Dimension')
    axes[0,1].set_ylabel('Position')
    
    # Compute and plot similarity matrices
    fixed_sim = torch.matmul(fixed_pos_enc, fixed_pos_enc.t())
    learned_sim = torch.matmul(learned_pos_enc.squeeze(0), learned_pos_enc.squeeze(0).t())
    
    sns.heatmap(fixed_sim.numpy(), ax=axes[1,0], cmap='viridis')
    axes[1,0].set_title('Fixed Positional Encoding Similarity')
    axes[1,0].set_xlabel('Position')
    axes[1,0].set_ylabel('Position')
    
    sns.heatmap(learned_sim.numpy(), ax=axes[1,1], cmap='viridis')
    axes[1,1].set_title('Learned Positional Encoding Similarity')
    axes[1,1].set_xlabel('Position')
    axes[1,1].set_ylabel('Position')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Model parameters
    image_size = (32, 32)
    patch_size = (4, 4)
    embed_dim = 128
    num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
    
    # Get fixed positional encoding
    fixed_pos_enc = positional_encoding_2d(
        nph=image_size[0] // patch_size[0],
        npw=image_size[1] // patch_size[1],
        dim=embed_dim
    )
    
    # Load trained model to get learned positional encoding
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        channels=3,
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=4,
        num_classes=2,
        pos_enc='learnable'
    )
    
    # Load the trained weights
    checkpoint = torch.load('model.pth', map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    
    # Get learned positional encoding
    learned_pos_enc = model.positional_embedding.detach()
    
    # Remove CLS token position if it exists
    if model.pool == 'cls':
        learned_pos_enc = learned_pos_enc[:, 1:, :]
    
    # Plot the encodings and their similarities
    plot_positional_encodings(fixed_pos_enc, learned_pos_enc, 'positional_encodings.png')