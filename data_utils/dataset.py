"""
PyTorch Dataset for Cholec80 surgical phase recognition.

Goal: Link frame images → frame_idx → phase labels
"""

import os
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader
import cv2


# Directory paths
#TEMP_FRAMES_DIR = os.path.expanduser("~/projects/surgical-phase-recognition/data/cholec80/frames")
FRAMES_DIR = os.path.expanduser("~surgical-phase-recognition/data/cholec80/frames")


class Cholec80Dataset:
    """
    Dataset that takes samples from dataloader.py and loads corresponding frame images.

    Input: samples = [(video_id, frame_idx, phase_label), ...]
    Output: (image, label_int) pairs
    """

    def __init__(self, samples, frames_dir=FRAMES_DIR, transform=None):
        """
        Store samples, frames directory, and optional transforms.
        Build a phase_label -> int mapping.
        """
        # parameters init
        self.samples = samples
        self.frames_dir = frames_dir
        self.transform = transform
        # phase mapping init
        self.phase2int = self._build_phase_mapping(samples)
        self.int2phase = {v: k for k, v in self.phase2int.items()}

    def __len__(self):
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load an image and its label from the dataset.
        Args:
            idx (int): index of sample
        Returns:
            image (Tensor): image tensor
            label_int (int): label integer
        """
        # First we want to extract key details from the sample
        video_id, frame_idx, phase_label = self.samples[idx]
        # Build image path from the video id and frame index
        img_path = os.path.join(self.frames_dir, video_id, f"frame_{frame_idx:04d}.jpg")
        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at: {img_path}")
        
        # Convert to RGB for torch + matplotlib compatibility
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply any transforms if provided
        if self.transform:
            img = self.transform(img)
        # Convert phase label to integer using our mapping (important to do this for the loss function)
        label_int = self.phase2int[phase_label]
        return img, label_int

    def _build_phase_mapping(self, samples):
        """
        Extract unique phase labels, sort them, create {phase: int} mapping.
        Args:
            samples (list): list of samples
        Returns:
            phase2int (dict): phase label to int mapping
                example: {"Pre-Grasp": 0, "Grasp": 1, ...}
        """
        # alphabetically sorted unique phase labels
        unique_phases = sorted(list(set(s[2] for s in samples)))
        # creating the mapping (returns each phase label)
        return {phase: i for i, phase in enumerate(unique_phases)}
