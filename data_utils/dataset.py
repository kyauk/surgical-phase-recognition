"""
PyTorch Dataset for Cholec80 surgical phase recognition.

Goal: Link frame images → frame_idx → phase labels
"""

import os
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
import cv2


# Directory paths
TEMP_FRAMES_DIR = os.path.expanduser("~/projects/surgical-phase-recognition/data/cholec80/frames")
FRAMES_DIR = os.path.expanduser("~surgical-phase-recognition/data/cholec80/frames")


class Cholec80Dataset(Dataset):
    """
    Dataset that takes samples from dataloader.py and loads corresponding frame images.

    Input: 
        video_frames: dict {video_id: [(frame_idx, phase_label), ...]}
        sequences: list [(video_id, start_idx), ...]
    Output: 
        images: Tensor (seq_len, C, H, W)
        labels: Tensor (seq_len,)
    """

    def __init__(self, video_frames, sequences, frames_dir=TEMP_FRAMES_DIR, transform=None, seq_len=16):
        """
        Store samples, frames directory, and optional transforms.
        Build a phase_label -> int mapping.
        """
        # parameters init
        self.video_frames = video_frames
        self.sequences = sequences
        self.frames_dir = frames_dir
        self.transform = transform
        self.seq_len = seq_len
        
        # phase mapping init
        self.phase2int = self._build_phase_mapping(video_frames)
        self.int2phase = {v: k for k, v in self.phase2int.items()}

    def __len__(self):
        """Return number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Load a sequence of images and their labels.
        Args:
            idx (int): index of sequence
        Returns:
            images (Tensor): image tensor of shape (seq_len, C, H, W)
            labels (Tensor): label tensor of shape (seq_len,)
        """
        # Get sequence info
        video_id, start_idx = self.sequences[idx]
        
        # storage
        seq_images = []
        seq_labels = []

        # Loop through the sequence
        for i in range(self.seq_len):
            current_seq_idx = start_idx + i
            # Get frame info from video_frames
            # video_frames[video_id] is a list/tuple of (frame_idx, phase_label)
            frame_idx, phase_label = self.video_frames[video_id][current_seq_idx]
            
            # Build image path
            img_path = os.path.join(self.frames_dir, video_id, f"frame_{frame_idx:04d}.jpg")
            
            # Load the image
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found at: {img_path}")
            
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply transforms
            if self.transform:
                img = self.transform(img)
            else:
                # If no transform, convert to tensor manually (H,W,C) -> (C,H,W) and normalize 0-1
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            # Convert phase label to int
            label_int = self.phase2int[phase_label]

            seq_images.append(img)
            seq_labels.append(label_int)

        # Stack into tensors
        # images: List[(C, H, W)] -> (seq_len, C, H, W)
        images = torch.stack(seq_images)
        # labels: List[int] -> (seq_len,)
        labels = torch.tensor(seq_labels, dtype=torch.long)
        
        return images, labels

    def _build_phase_mapping(self, video_frames):
        """
        Extract unique phase labels, sort them, create {phase: int} mapping.
        Args:
            video_frames (dict): dictionary of video samples
        Returns:
            phase2int (dict): phase label to int mapping
                example: {"Pre-Grasp": 0, "Grasp": 1, ...}
        """
        unique_phases = set()
        for frames in video_frames.values():
            for _, phase in frames:
                unique_phases.add(phase)
        
        # alphabetically sorted unique phase labels
        sorted_phases = sorted(list(unique_phases))
        
        # creating the mapping
        return {phase: i for i, phase in enumerate(sorted_phases)}
