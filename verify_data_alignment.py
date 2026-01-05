import torch
import os
import sys
import matplotlib.pyplot as plt
import torchvision
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_utils.dataloader import get_dataloaders
from constants import BATCH_SIZE, ANNOTATIONS_DIR

def denormalize(tensor):
    """Reverses the ImageNet normalization for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def verify_alignment():
    print("Initializing Dataloaders...")
    train_loader, _, _ = get_dataloaders(batch_size=4)
    
    print("Fetching first batch...")
    # Get one batch
    images, labels = next(iter(train_loader))
    
    # images: (B, Seq, C, H, W)
    # labels: (B, Seq)
    
    B, Seq, C, H, W = images.shape
    
    # Int to Phase mapping (reverse engineering or fetching from dataset if accessible)
    # We can access via the dataset object in the loader
    dataset = train_loader.dataset
    int2phase = dataset.int2phase
    
    print(f"Batch Shape: {images.shape}")
    print(f"Labels Shape: {labels.shape}")
    
    # Visualize the first sequence in the batch
    b_idx = 0
    seq_images = images[b_idx] # (Seq, C, H, W)
    seq_labels = labels[b_idx] # (Seq,)
    
    print(f"\nVisualizing Sequence {b_idx}...")
    
    # Denormalize
    seq_images_denorm = [denormalize(img).permute(1, 2, 0).numpy() for img in seq_images]
    
    # Create grid
    plt.figure(figsize=(20, 4))
    for i in range(min(Seq, 8)): # Show first 8 frames
        plt.subplot(1, 8, i+1)
        plt.imshow(np.clip(seq_images_denorm[i], 0, 1))
        
        label_id = seq_labels[i].item()
        label_name = int2phase[label_id]
        
        plt.title(f"{label_name}\n({label_id})")
        plt.axis('off')
        
    output_path = "debug_alignment.png"
    plt.savefig(output_path)
    print(f"\nSaved visualization to {output_path}")
    print("Please check this image to verify if the frame content matches the label.")

if __name__ == "__main__":
    verify_alignment()
