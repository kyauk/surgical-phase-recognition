
import os
import sys
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# Import project components
from models.model import SurgicalModel
from constants import BASE_DIR, SEQ_LEN

def run_inference(frames_dir, model_path, output_txt):
    """
    Run inference on a directory of frames for a single video.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Inference using device: {device}")

    # 1. Initialize Model
    model = SurgicalModel()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from {model_path}")
    else:
        print(f"ERROR: Model weights not found at {model_path}")
        return

    model.to(device)
    model.eval()

    # 2. Prepare Data Transform
    # Same normalization as training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Load Frames
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    if not frame_files:
        print(f"ERROR: No .jpg frames found in {frames_dir}")
        return

    print(f"Processing {len(frame_files)} frames...")

    # 4. Sliding Window / Sequence Processing
    # We process in sequences of SEQ_LEN. Note: Cholec80 is labeled per frame.
    # To keep it simple, we can slide with stride 1 or just non-overlapping.
    # Training used STRIDE for sampling, but inference usually runs on every frame 
    # and averages or just uses the most recent prediction.
    # For now, let's implement a simple sequence-based prediction.
    
    predictions = []
    
    # We'll use a sliding window with stride 1 to get a prediction for every frame once the window is full
    # For the first SEQ_LEN-1 frames, we can either pad or just start from SEQ_LEN.
    # Cholec80 evaluations often skip the first bit or pad with first frame.
    
    all_frame_tensors = []
    print("Pre-processing frames...")
    for f in frame_files:
        img_path = os.path.join(frames_dir, f)
        img = Image.open(img_path).convert('RGB')
        all_frame_tensors.append(transform(img))
    
    all_frames = torch.stack(all_frame_tensors) # (Total_Frames, C, H, W)
    
    print("Running Model...")
    with torch.no_grad():
        # Process in batches of sequences to speed up
        # Shape needed: (Batch, SEQ_LEN, C, H, W)
        
        # For simplicity, let's predict in windows
        for i in range(len(all_frames)):
            if i < SEQ_LEN:
                # Pad with the first frame for the beginning
                window = all_frames[0:SEQ_LEN].unsqueeze(0).to(device)
            else:
                window = all_frames[i-SEQ_LEN+1 : i+1].unsqueeze(0).to(device)
            
            output = model(window) # (1, SEQ_LEN, Num_Classes)
            # Take the prediction for the last frame in the sequence
            _, pred = output[0, -1].max(0)
            predictions.append(pred.item())

    # 5. Save Results
    # Cholec80 format usually: Frame\tPhase
    # (Note: Phase names would be better, but let's stick to indices if mapping is unknown, 
    # or better, fetch the mapping if possible. For inference, indices are safer unless mapping is provided.)
    
    with open(output_txt, 'w') as f:
        f.write("Frame\tPhase\n")
        for i, p in enumerate(predictions):
            f.write(f"{i}\t{p}\n")

    print(f"Done! Predictions saved to {output_txt}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on a video (frames folder).")
    parser.add_argument("--frames", type=str, required=True, help="Path to folder containing .jpg frames")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth", help="Path to best_model.pth")
    parser.add_argument("--output", type=str, default="predictions.txt", help="Path to output .txt file")
    
    args = parser.parse_args()
    
    # Resolve relative paths
    frames_path = os.path.abspath(args.frames)
    model_path = os.path.abspath(args.model)
    output_path = os.path.abspath(args.output)
    
    run_inference(frames_path, model_path, output_path)
