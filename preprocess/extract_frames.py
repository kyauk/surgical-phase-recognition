"""
Goal: take a mp4 video and turn it to frame by frame jpgs.

Input is a video
Extract ALL frames and save as JPGs
The dataloader will handle sampling every 12th frame and associating with labels
"""

import cv2
import os

TEMP_VIDEO_DIR = os.path.expanduser("~/projects/surgical-phase-recognition/data/cholec80/videos")
TEMP_FRAMES_DIR = os.path.expanduser("~/projects/surgical-phase-recognition/data/cholec80/frames")
VIDEO_DIR = os.path.expanduser("~/data/cholec80/videos")
FRAMES_DIR = os.path.expanduser("~/data/cholec80/frames")


def extract_frames(video_path, output_dir):
    """
    Extract all frames from a video and save as JPGs.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video {video_path}")
        return

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Extracting frames from {os.path.basename(video_path)}...")
    print(f"Total frames: {total_frames}")

    while True:
        ret, frame = cap.read()
        # If not successfully read
        if not ret:
            break

        # Save frame as JPG with zero-padded filename
        frame_filename = f"frame_{frame_idx:04d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        frame_idx += 1

        # Progress update every 100 frames
        if frame_idx % 100 == 0:
            print(f"  Extracted {frame_idx}/{total_frames} % frames...")

    cap.release()
    print(f"Done! Extracted {frame_idx} frames to {output_dir}\n")


def extract_all_videos(video_dir, frames_dir):
    """
    Extract frames from all videos in a directory.

    Args:
        video_dir: Directory containing video files
        frames_dir: Base directory to save extracted frames
    """
    # Get all video files
    video_files = [
        f for f in os.listdir(video_dir)
        if f.endswith(('.mp4', '.avi', '.mov'))
    ]

    if len(video_files) == 0:
        print(f"No video files found in {video_dir}")
        return

    print(f"Found {len(video_files)} videos to process\n")

    for video_file in sorted(video_files):
        # Get video ID (e.g., "video01" from "video01.mp4")
        video_id = os.path.splitext(video_file)[0]

        # Create output directory for this video
        video_output_dir = os.path.join(frames_dir, video_id)

        # Skip if already extracted
        if os.path.exists(video_output_dir) and len(os.listdir(video_output_dir)) > 0:
            print(f"Skipping {video_file} (already extracted)")
            continue

        # Extract frames
        video_path = os.path.join(video_dir, video_file)
        extract_frames(video_path, video_output_dir)


def main():
    # Extract frames from all videos
    extract_all_videos(TEMP_VIDEO_DIR, TEMP_FRAMES_DIR)


if __name__ == "__main__":
    main()
