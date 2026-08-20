"""
scripts/list_holdout_videos.py
Reproduce el mismo split 50/50 (finetune/holdout) que uso 04_finetune_peru.py
y lista los nombres de archivo del holdout -- para elegir clips para el video demo.
"""

import pandas as pd
import numpy as np
from pathlib import Path

CLASSES = ["Robbery", "Burglary", "Shoplifting", "Stealing", "Normal"]
SEED = 42


def split_peru_finetune_holdout(peru_manifest: pd.DataFrame, classes: list, seed: int = SEED):
    rng = np.random.default_rng(seed)
    finetune_rows, holdout_rows = [], []

    for clase in classes:
        subset = peru_manifest[peru_manifest["clase"] == clase]
        if len(subset) == 0:
            continue
        idx = subset.index.to_numpy()
        rng.shuffle(idx)
        half = len(idx) // 2
        finetune_rows.append(peru_manifest.loc[idx[:half]])
        holdout_rows.append(peru_manifest.loc[idx[half:]])

    finetune_df = pd.concat(finetune_rows, ignore_index=True)
    holdout_df = pd.concat(holdout_rows, ignore_index=True)
    return finetune_df, holdout_df


if __name__ == "__main__":
    manifest = pd.read_csv("data/processed/manifest.csv")
    peru_manifest = manifest[manifest["split"] == "peru_eval"]

    finetune_df, holdout_df = split_peru_finetune_holdout(peru_manifest, CLASSES)

    print("=== HOLDOUT (nunca visto en fine-tuning, usalos para la demo/video) ===")
    for clase in ["Robbery", "Normal"]:
        videos = sorted(holdout_df[holdout_df["clase"] == clase]["video_id"].tolist())
        print(f"\n{clase} ({len(videos)} videos):")
        for v in videos:
            print(f"  {v}.mp4")