"""
Quick sanity check: counts images per class across the MSLD v2.0 fold1 splits.
"""

import os

BASE = "data/raw/MSLD_v2/Original Images/Original Images/FOLDS/fold1"

for split in ["Train", "Valid", "Test"]:
    split_path = os.path.join(BASE, split)
    print(f"\n{split}:")
    if not os.path.isdir(split_path):
        print(f"  MISSING: {split_path}")
        continue
    total = 0
    for cls in sorted(os.listdir(split_path)):
        cls_path = os.path.join(split_path, cls)
        if os.path.isdir(cls_path):
            n = len([f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            print(f"  {cls}: {n}")
            total += n
    print(f"  TOTAL: {total}")
