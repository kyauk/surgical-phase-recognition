
"""
Global Constants for the Surgical Phase Recognition Project.
"""

# Data Processing
STRIDE = 12         # Sampling rate for frames
SEQ_LEN = 16        # Sequence length for LSTM/Transformer input
FRAME_HEIGHT = 224
FRAME_WIDTH = 224

# Model (Default info)
NUM_CLASSES = 7
BATCH_SIZE = 4

import os

# Base Paths (Switch these based on environment)
# Use os.path.expanduser to handle "~"
LOCAL_BASE_DIR = os.path.expanduser("~/projects/surgical-phase-recognition")
CLOUD_BASE_DIR = os.path.expanduser("~/surgical-phase-recognition")

# CHANGE THIS to switch between Local and Cloud
BASE_DIR = CLOUD_BASE_DIR

# Derived Data Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "phase-annotations")

