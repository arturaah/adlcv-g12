import torch
import matplotlib.pyplot as plt
from einops import rearrange
import torchvision.transforms as transforms
from vit import ViT
from imageclassification import prepare_dataloaders, set_seed

def get_attention_maps(model, image, head_idx=0):
    """Extracts attention maps from the model for a given image."""
    model.eval()
    attention_maps = []
    
    def hook_fn(module, input, output):
        # Extract attention weights from the attention computation
        batch_size, seq_len, _ = input[0].shape
        q = module.q_projection(input[0])
        k = module.k_projection(input[0])
        
        q = rearrange(q, 'b s (h d) -> b h s d', h=module.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=module.num_heads)
        
        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * module.scale
        attention = dots.softmax(dim=-1)
        
        # Extract the attention weights for the specified head
        attention_maps.append(attention[0, head_idx].detach().cpu())
    
    # Register hooks for each transformer block
    hooks = []
    for block in model.transformer_blocks:
        hooks.append(block.attention.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        model(image)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_maps

def plot_attention_maps(image, attention_maps, patch_size=(4, 4)):
    """Visualizes attention maps for each transformer layer."""
    num_layers = len(attention_maps)
    fig, axes = plt.subplots(num_layers, 3, figsize=(15, 5*num_layers))
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    # Plot original image
    img = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    
    # Plot attention maps for each layer
    for idx, attn in enumerate(attention_maps):
        # Plot original image
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f'Layer {idx+1}\nOriginal Image')
        axes[idx, 0].axis('off')
        
        # Get attention weights for the CLS token
        cls_attn = attn[0, 1:].reshape(image.shape[2]//patch_size[0], 
                                      image.shape[3]//patch_size[1])
        
        # Plot raw attention map
        axes[idx, 1].imshow(cls_attn, cmap='viridis')
        axes[idx, 1].set_title(f'Layer {idx+1}\nAttention Map')
        axes[idx, 1].axis('off')
        
        # Plot attention overlay
        axes[idx, 2].imshow(img)
        axes[idx, 2].imshow(cls_attn, cmap='viridis', alpha=0.5)
        axes[idx, 2].set_title(f'Layer {idx+1}\nOverlay')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set device and seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed=1)
    
    # Load model with correct configuration
    model = ViT(
        image_size=(32, 32),
        patch_size=(4, 4),
        channels=3,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        num_classes=2,
        pos_enc='learnable',
        pool='cls'
    ).to(device)
    
    try:
        # Load trained weights with safety flag
        checkpoint = torch.load('model.pth', map_location=device, weights_only=True)
        
        # Handle potential state dict mismatches
        model_state_dict = model.state_dict()
        for key in list(checkpoint.keys()):
            if key not in model_state_dict:
                print(f"Removing key {key} from loaded state dict")
                del checkpoint[key]
        
        model.load_state_dict(checkpoint, strict=False)
        print("Model loaded successfully")
        
        # Get sample image
        _, _, _, testset = prepare_dataloaders(batch_size=1)
        image, label = testset[0]
        image = image.unsqueeze(0).to(device)
        
        # Get and visualize attention maps
        attention_maps = get_attention_maps(model, image)
        plot_attention_maps(image, attention_maps)
        
    except Exception as e:
        print(f"Error: {e}")
        exit(1)