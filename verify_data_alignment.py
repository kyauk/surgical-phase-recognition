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

import random

def verify_metadata():
    print("Initializing Dataloaders...")
    # using batch_size 1 is fine, we just need the dataset
    train_loader, _, _ = get_dataloaders(batch_size=1)
    dataset = train_loader.dataset
    
    if len(dataset) == 0:
        print("Dataset is empty! Check your data paths.")
        return

    # Pick a random sequence index
    idx = random.randint(0, len(dataset) - 1)
    
    # Get sequence info directly from dataset internal storage
    video_id, start_idx = dataset.sequences[idx]
    
    print(f"\n--- Verification Sample (Index {idx}) ---")
    print(f"Video ID: {video_id}")
    print(f"Sequence Start Index (internal): {start_idx}")
    
    # Find the correct annotation file
    # In dataloader, video_id was created via: video.replace("-", ".").split(".")[0]
    # We need to find the filename that contains this video_id
    try:
        annotation_files = os.listdir(ANNOTATIONS_DIR)
        target_file = None
        for f in annotation_files:
            if video_id in f:
                target_file = f
                break
    except FileNotFoundError:
        print(f"Error: ANNOTATIONS_DIR not found at {ANNOTATIONS_DIR}")
        return
            
    if target_file:
        full_path = os.path.join(ANNOTATIONS_DIR, target_file)
        print(f"\nAnnotation File: {full_path}")
        
        print(f"\n--- Sequence Details (Length {dataset.seq_len}) ---")
        # Get the frames in the sequence
        first_frame = None
        last_frame = None
        
        for i in range(dataset.seq_len):
            current_seq_idx = start_idx + i
            # dataset.video_frames is {video_id: [(frame_idx, phase), ...]}
            frame_idx, phase_label = dataset.video_frames[video_id][current_seq_idx]
            
            print(f"Step {i:02d}: Frame {frame_idx:<6} | Label: {phase_label}")
            
            if i == 0: first_frame = frame_idx
            if i == dataset.seq_len - 1: last_frame = frame_idx
            
        print(f"\n--- Manual Verification Command ---")
        print(f"Run this to check the original text file:")
        print(f"grep -C 5 '^{first_frame}\\b' {full_path}")
        
    else:
        print(f"Could not find annotation file for video {video_id} in {ANNOTATIONS_DIR}")

if __name__ == "__main__":
    verify_metadata()
