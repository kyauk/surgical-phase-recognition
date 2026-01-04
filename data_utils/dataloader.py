"""
Phase annotations:
# frame_idx, phase

path: ~/data/cholec80/phase_annotations/

use temp phase dir when doing work localy, phase dir for cloud work
"""

import random
from torch.utils.data import DataLoader
from .dataset import Cholec80Dataset
from torchvision import transforms
import sys
import os

# Add project root to path to find constants.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import STRIDE, SEQ_LEN, ANNOTATIONS_DIR

SEED = 42
# TEMP_PHASE_DIR and PHASE_DIR removed in favor of single ANNOTATIONS_DIR from constants


def get_videos_data(path):
    """
    Return true number of videos that are annotated
    Args:
        path (str): path to the annotated files
    Returns:
        video_files (list): list of video files
        num_videos (int): number of videos
    """
    video_files = [
        f for f in os.listdir(path)
        if f.endswith(".txt")
    ]
    num_videos = len(video_files)
    return video_files, num_videos

def split_videos(video_files, num_videos):
    """
    Split videos into Train/Val/Test (60/20/20 percents)
    Args:
        video_files (list): list of video files
        num_videos (int): number of videos
    Returns:
        train_videos (list): list of train videos
        val_videos (list): list of validation videos
        test_videos (list): list of test videos
    """
    random.seed(SEED)
    # get split indices
    train_num = int(num_videos * 0.6)
    val_num = int(num_videos * 0.2)
    val_idx = train_num + val_num
    # random shuffle
    video_files = video_files.copy()
    random.shuffle(video_files)
    # assign split types
    train_videos = video_files[:train_num]
    val_videos = video_files[train_num:val_idx]
    test_videos= video_files[val_idx:]

    return train_videos, val_videos, test_videos


def load_annotations(annotated_file_path):
    """
    Load phase annotations
    want to return, for a single annotated .txt file, a list of tuples (frame_idx, phase_label).
    Args:
        annotated_file_path (str): path to the annotated file
    Returns:
        annotations (list): list of tuples (frame_idx, phase_label)
    """
    # load the annotated file
    annotations = []
    # go through line by line, and parse between frame index and annotated phase label
    with open(annotated_file_path, "r") as f:
        for i, line in enumerate(f):
            # skip header line
            if i == 0:
                continue
            frame_idx, phase_label = line.strip().split()
            annotations.append((int(frame_idx), phase_label))
    return annotations

def get_all_unique_phases(annotated_path, video_files):
    """
    Scan all video files to find every unique phase label in the entire dataset.
    Ensures consistency across splits.
    """
    unique_phases = set()
    for video_file in video_files:
        path = os.path.join(annotated_path, video_file)
        annotations = load_annotations(path)
        for _, phase in annotations:
            unique_phases.add(phase)
    
    sorted_phases = sorted(list(unique_phases))
    return {phase: i for i, phase in enumerate(sorted_phases)}



def build_samples(videos: list, annotated_path: str, stride=STRIDE,seq_len=SEQ_LEN):
    """
    Build samples where any split type works

    Args:
        videos (list): list of video files
        annotated_path (str): path to annotated files
        stride (int): stride between frames
        seq_len (int): sequence length
    Returns:
        samples (dict): dictionary of string keys, values of tuple of tuples values
        sequences (list): list of tuples (video_id, start_idx) to be used for __getitem__ in Cholec80Dataset
    """

    samples = {}
    for video in videos:
        # split by dot or dash to get "video01" from "video01-phase.txt"
        video_id = video.replace("-", ".").split(".")[0]
        video_path = os.path.join(annotated_path, video)
        annotations = load_annotations(video_path)
        frames = []

        for frame_idx, phase_label in annotations:
            if frame_idx % STRIDE == 0:
                frames.append((frame_idx, phase_label))

        samples[video_id] = tuple(frames)
            
    sequences = []
    for video, frames in samples.items():
        num_frames = len(frames)
        max_start_idx = num_frames - seq_len
        i = 0
        for i in range(max_start_idx + 1):
            sequences.append((video, i))
    return samples, sequences

def get_dataloaders(annotated_path=ANNOTATIONS_DIR, batch_size=32):
    """
    Helper function create train/val/test dataloaders
    Args:
        annotated_path (str): path to annotated files
    Returns:
        train_loader (DataLoader): train dataloader
        val_loader (DataLoader): validation dataloader
        test_loader (DataLoader): test dataloader
    """
    # first get video files and split them
    video_files, num_videos = get_videos_data(annotated_path)
    train_videos, val_videos, test_videos = split_videos(video_files, num_videos)

    # 1. Create Global Phase Mapping
    phase_mapping = get_all_unique_phases(annotated_path, video_files)
    print(f"Global Phase Mapping: {phase_mapping}")

    # 2. Define Transforms (ImageNet Normalization)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # build samples for each split
    train_frames, train_sequences = build_samples(train_videos, annotated_path)
    val_frames, val_sequences = build_samples(val_videos, annotated_path)
    test_frames, test_sequences = build_samples(test_videos, annotated_path)

    # create datasets
    train_dataset = Cholec80Dataset(train_frames, train_sequences, transform=data_transform, phase_mapping=phase_mapping)
    val_dataset = Cholec80Dataset(val_frames, val_sequences, transform=data_transform, phase_mapping=phase_mapping)
    test_dataset = Cholec80Dataset(test_frames, test_sequences, transform=data_transform, phase_mapping=phase_mapping)

    # create dataloaders
    # create dataloaders
    train_loader = []
    if len(train_dataset) > 0:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    val_loader = []
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
    test_loader = []
    if len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader
