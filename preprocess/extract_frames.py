"""
Goal: take a mp4 video and turn it to frame by frame jpgs.

Input is a video
Extract ALL frames and save as JPGs
The dataloader will handle sampling every 12th frame and associating with labels
"""

import cv2
import os
from tqdm import tqdm

#TEMP_VIDEO_DIR = os.path.expanduser("~/projects/surgical-phase-recognition/data/cholec80/videos")
#TEMP_FRAMES_DIR = os.path.expanduser("~/projects/surgical-phase-recognition/data/cholec80/frames")
VIDEO_DIR = os.path.expanduser("~/surgical-phase-recognition/data/cholec80/videos")
FRAMES_DIR = os.path.expanduser("~/surgical-phase-recognition/data/cholec80/frames")


def extract_frames(video_path, output_dir):
    """
    Extract all frames from a video and save as JPGs.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.basename(video_path)

    frame_idx = 0
    
    # Use tqdm for frame extraction
    with tqdm(total=total_frames, desc=f"  {video_name}", unit="fr", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = f"frame_{frame_idx:04d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

            frame_idx += 1
            pbar.update(1)

    cap.release()

def extract_all_videos(video_dir, frames_dir):
    """
    Extract frames from all videos in a directory.
    """
    video_files = [
        f for f in os.listdir(video_dir)
        if f.endswith(('.mp4', '.avi', '.mov'))
    ]

    if len(video_files) == 0:
        print(f"No video files found in {video_dir}")
        return

    print(f"Found {len(video_files)} videos to process")

    # Use tqdm for video loop
    for video_file in tqdm(sorted(video_files), desc="Total Progress", unit="vid"):
        video_id = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(frames_dir, video_id)

        if os.path.exists(video_output_dir) and len(os.listdir(video_output_dir)) > 0:
            continue

        video_path = os.path.join(video_dir, video_file)
        extract_frames(video_path, video_output_dir)


def main():
    # Extract frames from all videos
    extract_all_videos(VIDEO_DIR, FRAMES_DIR)


if __name__ == "__main__":
    main()
