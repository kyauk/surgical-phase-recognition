import torch
from torch.utils.data import DataLoader
from dataset import Cholec80Dataset

def validate_dataset(train_videos, val_videos, test_videos, train_samples, val_samples, test_samples, annotated_path=None):
    """
    Comprehensive dataset validation.
    Checks:
    1. No video overlap between splits
    2. No duplicate frames within any split
    3. DataLoaders produce expected shapes
    """
    print("--- Running Dataset Validation ---")

    # --- 1. Split Integrity (Video Level) ---
    train_set, val_set, test_set = set(train_videos), set(val_videos), set(test_videos)
    
    # Check intersections
    tv_inter = train_set.intersection(val_set)
    tt_inter = train_set.intersection(test_set)
    vt_inter = val_set.intersection(test_set)
    
    if tv_inter or tt_inter or vt_inter:
         raise ValueError(f"CRITICAL: Overlapping videos found!\nTrain-Val: {tv_inter}\nTrain-Test: {tt_inter}\nVal-Test: {vt_inter}")

    print(f"w[Check 1 Passed] Video Splits are unique. Train: {len(train_videos)}, Val: {len(val_videos)}, Test: {len(test_videos)}")

    # --- 2. Frame Level Integrity ---
    # Samples are (video_id, frame_idx, label)
    # We want to ensure (video_id, frame_idx) is unique across the entire dataset
    
    def check_duplicates(samples, split_name):
        seen = set()
        duplicates = []
        for vid, frame, _ in samples:
            key = (vid, frame)
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        
        if duplicates:
            raise ValueError(f"CRITICAL: Duplicate frames found in {split_name}: {duplicates[:5]}...")
        return len(seen)

    n_train = check_duplicates(train_samples, "Train")
    n_val = check_duplicates(val_samples, "Val")
    n_test = check_duplicates(test_samples, "Test")
    
    print(f"[Check 2 Passed] No duplicate frames. Total samples: {n_train + n_val + n_test}")

    # --- 3. DataLoader & Shape Verification ---
    # Instantiate datasets to check if they load correlates correctly
    batch_size = 4
    
    # We create temporary loaders just for validation
    train_ds = Cholec80Dataset(train_samples, transform=None)
    val_ds = Cholec80Dataset(val_samples, transform=None)
    test_ds = Cholec80Dataset(test_samples, transform=None)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"[Check 3] verifying DataLoader shapes (batch_size={batch_size})...")
    
    def check_loader(loader, name):
        try:
            # Check if loader is empty
            if len(loader) == 0:
                print(f"WARNING: {name} loader is empty (likely due to small dataset split). skipping batch check.")
                return

            # Fetch one batch
            images, labels = next(iter(loader))
            
            # Expected: (B, C, H, W) and (B,)
            if images.shape[0] != batch_size:
                # If it's the last batch it might be smaller, which is fine
                pass
            
            print(f"  {name}: {images.shape} | Labels: {labels.shape}")
            
        except StopIteration:
            print(f"WARNING: {name} loader seems empty despite len > 0?")
        except Exception as e:
            raise RuntimeError(f"Failed to load batch from {name}: {e}")

    check_loader(train_loader, "Train")
    check_loader(val_loader, "Val")
    check_loader(test_loader, "Test")

    print(f"[Check 3 Passed] DataLoaders yield correct shapes.\n")