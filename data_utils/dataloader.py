"""
Phase annotations:
# frame_idx, phase

path: ~/data/cholec80/phase_annotations/

use temp phase dir when doing work localy, phase dir for cloud work
"""

import os
import random
from torch.utils.data import DataLoader
from dataset import Cholec80Dataset
from validate_dataset import validate_dataset

STRIDE = 12
SEED = 42
#TEMP_PHASE_DIR = os.path.expanduser("~/projects/surgical-phase-recognition/data/cholec80/phase_annotations")
PHASE_DIR = os.path.expanduser("~/data/cholec80/phase_annotations")

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

def build_samples(videos, annotated_path, stride=STRIDE):
    """
    Build samples where any split type works

    1. for each video, get its associated .txt file, and load their annotations
    2. make tuple of associated
    """
    samples = []
    for video in videos:
        # Robustly split by dot or dash to get "video01" from "video01-phase.txt"
        video_id = video.replace("-", ".").split(".")[0]
        
        video_path = os.path.join(annotated_path, video)
        annotations = load_annotations(video_path)
        for frame_idx, phase_label in annotations:
            # every 12th frame is where we want to sample
            if frame_idx % STRIDE == 0:
                samples.append((video_id, frame_idx, phase_label))
    return samples

# Helper function to create train/val/test dataloaders
def get_dataloaders(annotated_path):
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

    # build samples for each split
    train_samples = build_samples(train_videos, annotated_path)
    val_samples = build_samples(val_videos, annotated_path)
    test_samples = build_samples(test_videos, annotated_path)
    
    # Run validation check
    validate_dataset(train_videos, val_videos, test_videos, 
                     train_samples, val_samples, test_samples, 
                     annotated_path)

    # create datasets
    train_dataset = Cholec80Dataset(train_samples, transform=None)
    val_dataset = Cholec80Dataset(val_samples, transform=None)
    test_dataset = Cholec80Dataset(test_samples, transform=None)

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader




def main():
    print("Testing dataloader...")
    train_loader, val_loader, test_loader = get_dataloaders(PHASE_DIR)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Check one batch
    images, labels = next(iter(train_loader))
    print(f"\nSample batch shape: {images.shape}")
    print(f"Sample labels shape: {labels.shape}")
    print("Success!")


if __name__ == "__main__":
    main()
