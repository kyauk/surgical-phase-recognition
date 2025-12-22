"""
Phase annotations:
# frame_idx, phase

path: ~/data/cholec80/phase_annotations/

use temp phase dir when doing work localy, phase dir for cloud work
"""

import os
import random

STRIDE = 12
SEED = 42
TEMP_PHASE_DIR = os.path.expanduser("~/projects/surgical-phase-recognition/data/cholec80/phase_annotations")
PHASE_DIR = os.path.expanduser("~/data/cholec80/phase_annotations")

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
        video_id = os.path.splitext(video)[0]
        video_path = os.path.join(annotated_path, video)
        annotations = load_annotations(video_path)
        for frame_idx, phase_label in annotations:
            if frame_idx % STRIDE == 0:
                samples.append((video_id, frame_idx, phase_label))
    return samples

def validate_dataset(train_videos, val_videos, test_videos, train_samples, val_samples, test_samples, annotated_path):
    """
    Sanity checks to make sure dataset is properly split and formatted.
    """
    print("\nRunning dataset validation...\n")

    # check for overlapping videos between splits
    train_set = set(train_videos)
    val_set = set(val_videos)
    test_set = set(test_videos)

    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set

    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("ERROR: Found overlapping videos between splits!")
        if overlap_train_val:
            print(f"  Train/Val overlap: {overlap_train_val}")
        if overlap_train_test:
            print(f"  Train/Test overlap: {overlap_train_test}")
        if overlap_val_test:
            print(f"  Val/Test overlap: {overlap_val_test}")
        raise ValueError("Splits contain overlapping videos")

    print(f"No overlapping videos between splits")
    print(f"  Train: {len(train_videos)} videos")
    print(f"  Val: {len(val_videos)} videos")
    print(f"  Test: {len(test_videos)} videos\n")

    # validate annotation files and collect stats
    all_videos = train_videos + val_videos + test_videos
    all_phase_labels = set()
    video_stats = {}

    for video in all_videos:
        video_id = os.path.splitext(video)[0]
        video_path = os.path.join(annotated_path, video)

        try:
            annotations = load_annotations(video_path)
        except Exception as e:
            print(f"ERROR: Failed to load {video}: {e}")
            raise

        if len(annotations) == 0:
            print(f"WARNING: {video} has no annotations - skipping")
            continue

        frame_indices = [frame_idx for frame_idx, _ in annotations]
        phase_labels = [phase_label for _, phase_label in annotations]

        # Check for duplicate frame indices
        if len(frame_indices) != len(set(frame_indices)):
            print(f"WARNING: {video} has duplicate frame indices")

        # Check for negative frames
        if any(idx < 0 for idx in frame_indices):
            print(f"ERROR: {video} has negative frame indices")
            raise ValueError(f"Negative frame indices in {video}")

        # Check if sequential
        sorted_indices = sorted(frame_indices)
        if frame_indices != sorted_indices:
            print(f"WARNING: {video} frame indices are not in order")

        all_phase_labels.update(phase_labels)
        unique_phases = set(phase_labels)

        video_stats[video_id] = {
            'num_frames': len(annotations),
            'min_frame': min(frame_indices),
            'max_frame': max(frame_indices),
            'unique_phases': len(unique_phases),
            'phases': unique_phases
        }

    print(f"All {len(all_videos)} annotation files loaded successfully")
    print(f"  Found {len(all_phase_labels)} unique phases: {sorted(all_phase_labels)}\n")

    # Sample counts
    print(f"Sample counts:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    print(f"  Test: {len(test_samples)}")
    print(f"  Total: {len(train_samples) + len(val_samples) + len(test_samples)}\n")

    # Phase distribution across splits
    from collections import Counter

    def get_phase_distribution(samples):
        phases = [phase for _, _, phase in samples]
        return Counter(phases)

    train_dist = get_phase_distribution(train_samples)
    val_dist = get_phase_distribution(val_samples)
    test_dist = get_phase_distribution(test_samples)

    print("Phase distribution across splits:")
    print(f"  {'Phase':<15} {'Train':<10} {'Val':<10} {'Test':<10}")
    for phase in sorted(all_phase_labels):
        print(f"  {phase:<15} {train_dist.get(phase, 0):<10} {val_dist.get(phase, 0):<10} {test_dist.get(phase, 0):<10}")

    # Samples per video
    def get_samples_per_video(samples):
        video_ids = [video_id for video_id, _, _ in samples]
        return Counter(video_ids)

    train_per_video = get_samples_per_video(train_samples)
    val_per_video = get_samples_per_video(val_samples)
    test_per_video = get_samples_per_video(test_samples)

    all_samples_per_video = {}
    all_samples_per_video.update(train_per_video)
    all_samples_per_video.update(val_per_video)
    all_samples_per_video.update(test_per_video)

    sample_counts = list(all_samples_per_video.values())
    print(f"\nSamples per video: min={min(sample_counts)}, max={max(sample_counts)}, avg={sum(sample_counts)/len(sample_counts):.1f}")

    # Check for videos with suspiciously few samples
    min_expected_samples = 10
    low_sample_videos = [vid for vid, count in all_samples_per_video.items() if count < min_expected_samples]
    if low_sample_videos:
        print(f"WARNING: {len(low_sample_videos)} videos have fewer than {min_expected_samples} samples")

    # Check for single-phase videos
    single_phase_videos = [vid for vid, stats in video_stats.items() if stats['unique_phases'] == 1]
    if single_phase_videos:
        print(f"WARNING: {len(single_phase_videos)} videos have only one phase (might be annotation issue)")

    # Frame ranges
    frame_mins = [stats['min_frame'] for stats in video_stats.values()]
    frame_maxs = [stats['max_frame'] for stats in video_stats.values()]

    print(f"\nFrame ranges: min={min(frame_mins)}, max={max(frame_maxs)}")

    if min(frame_mins) != 0:
        print(f"WARNING: Some videos don't start at frame 0 (earliest is {min(frame_mins)})")

    print(f"\nValidation complete!\n")





def main():
    # first load video data
    video_files, num_videos = get_videos_data(TEMP_PHASE_DIR)
    train_videos, val_videos, test_videos = split_videos(video_files, num_videos)
    train_samples = build_samples(train_videos, TEMP_PHASE_DIR)
    val_samples = build_samples(val_videos, TEMP_PHASE_DIR)
    test_samples = build_samples(test_videos, TEMP_PHASE_DIR)

    # run validation
    validate_dataset(train_videos, val_videos, test_videos,
                     train_samples, val_samples, test_samples,
                     TEMP_PHASE_DIR)


if __name__ == "__main__":
    main()

# First load dataset
# Second split dataset into train, val, test
# Third make samples for each train,val,test (every 12th sample)
# Fourth do a sanity check
# Dataset: given an idx, return (x,y)
