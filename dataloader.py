""""
Dataloader
"""
"""
Phase annotations:
# frame_idx, phase


path: ~/data/cholec80/phase_annotations/
"""

import os
import random
temp_phase_dir = os.path.expanduser("~/projects/surgical-phase-recognition/data/cholec80/phase_annotations")
phase_dir = os.path.expanduser("~/data/cholec80/phase_annotations")

def get_videos_data(path):
    """
    Return true number of videos that are annotated
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
    """
    video_files, num_videos = get_videos_data()
    train_num = int(num_videos * 0.6)
    val_num = int(num_videos() * 0.2)
    val_idx = train_num + val_num
    random.shuffle(video_files)
    train_videos = video_files[:train_num]
    val_videos = video_files[train_num:val_idx]
    test_videos= video_files[val_idx:]

    videos = list(train_videos, val_videos, test_videos)
    return videos

def create_samples(video_splits):
    """
    videos = [train_videos, val_videos,test_videos]
    Creates sample tuples (train_examples, val_examples, test_examples)
    """
    for videos in video_splits:
        for video in videos:
            for frame in video:
                pass
            
    








def load_annotations():
    pass



class Dataset:
    pass


def main():
    get_num_videos()

if __name__ == "__main__":
    main()

# First load dataset
# Second split dataset into train, val, test
# Third make samples for each train,val,test (every 12th sample)
# Fourth do a sanity check
# Dataset: given an idx, return (x,y)
