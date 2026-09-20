"""
Merges MSLD v2.0 (dermatologist-verified) and MSID (web-sourced) image
datasets into one combined pool, then creates a fresh stratified
Train/Valid/Test split (70/15/15) across all 6 disease classes.

MSID's "Normal" class is mapped to "Healthy" to match MSLD's naming.
MSID has no Cowpox or HFMD classes, so those remain MSLD-only.

Provenance note: MSLD images are dermatologist-verified; MSID images are
collected from internet sources. Mixing them adds volume (especially for
the scarce Measles class) at some cost to label-quality certainty — this
is documented here so it can be cited honestly in the project report.
"""

import os
import shutil
import random
from collections import defaultdict

random.seed(42)

MSLD_BASE = "data/raw/MSLD_v2/Original Images/Original Images/FOLDS/fold1"
MSID_BASE = "data/raw/MSID/Monkeypox Skin Image Dataset"
OUTPUT_BASE = "data/processed/combined_images"

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

MSID_CLASS_MAP = {
    "Chickenpox": "Chickenpox",
    "Measles": "Measles",
    "Monkeypox": "Monkeypox",
    "Normal": "Healthy",
}

SPLIT_RATIOS = {"Train": 0.70, "Valid": 0.15, "Test": 0.15}


def collect_msld_images():
    """MSLD's 5 folds are just different splits of the same underlying
    755 images, so pooling fold1's Train+Valid+Test gives every unique
    image exactly once."""
    pool = defaultdict(list)
    for split in ["Train", "Valid", "Test"]:
        split_dir = os.path.join(MSLD_BASE, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    pool[cls].append(os.path.join(cls_dir, fname))
    return pool


def collect_msid_images():
    pool = defaultdict(list)
    if not os.path.isdir(MSID_BASE):
        print(f"WARNING: MSID not found at {MSID_BASE}, skipping.")
        return pool
    for msid_cls, unified_cls in MSID_CLASS_MAP.items():
        cls_dir = os.path.join(MSID_BASE, msid_cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(IMG_EXTENSIONS):
                pool[unified_cls].append(os.path.join(cls_dir, fname))
    return pool


def merge_pools(msld_pool, msid_pool):
    combined = defaultdict(list)
    for cls, paths in msld_pool.items():
        combined[cls].extend(paths)
    for cls, paths in msid_pool.items():
        combined[cls].extend(paths)
    return combined


def stratified_split_and_copy(combined_pool):
    if os.path.isdir(OUTPUT_BASE):
        shutil.rmtree(OUTPUT_BASE)

    summary = {}
    for cls, paths in combined_pool.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = int(n * SPLIT_RATIOS["Train"])
        n_valid = int(n * SPLIT_RATIOS["Valid"])
        splits = {
            "Train": paths[:n_train],
            "Valid": paths[n_train:n_train + n_valid],
            "Test": paths[n_train + n_valid:],
        }
        summary[cls] = {k: len(v) for k, v in splits.items()}
        summary[cls]["total"] = n

        for split_name, split_paths in splits.items():
            out_dir = os.path.join(OUTPUT_BASE, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)
            for i, src_path in enumerate(split_paths):
                ext = os.path.splitext(src_path)[1]
                dst_path = os.path.join(out_dir, f"{cls}_{split_name}_{i:04d}{ext}")
                shutil.copy2(src_path, dst_path)

    return summary


if __name__ == "__main__":
    print("Collecting MSLD images...")
    msld_pool = collect_msld_images()
    for cls, paths in msld_pool.items():
        print(f"  MSLD {cls}: {len(paths)}")

    print("\nCollecting MSID images...")
    msid_pool = collect_msid_images()
    for cls, paths in msid_pool.items():
        print(f"  MSID {cls}: {len(paths)}")

    print("\nMerging pools...")
    combined = merge_pools(msld_pool, msid_pool)

    print("\nSplitting and copying into", OUTPUT_BASE)
    summary = stratified_split_and_copy(combined)

    print("\n" + "=" * 60)
    print("FINAL COMBINED DATASET SUMMARY")
    print("=" * 60)
    for cls, counts in sorted(summary.items()):
        print(f"{cls:12s} — total: {counts['total']:4d}  "
              f"(Train: {counts['Train']}, Valid: {counts['Valid']}, Test: {counts['Test']})")
