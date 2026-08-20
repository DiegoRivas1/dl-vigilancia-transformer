"""
scripts/augment_train.py
Genera copias aumentadas de las secuencias de TRAIN (nunca de val/peru_eval)
para combatir el overfitting por pocos datos. Aplica la misma transformacion
a toda la secuencia de un video (coherencia temporal).
"""

import cv2
import numpy as np
import pandas as pd
import random
from pathlib import Path

N_AUGMENTED_COPIES = 2  # por video de train -> triplica el train set


def augment_sequence(sequence: np.ndarray, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    seq = sequence.copy()

    #if rng.random() < 0.5:
    #    seq = seq[:, :, ::-1, :]  # flip horizontal, mismo para todos los frames

    brightness = rng.uniform(0.8, 1.2)
    contrast = rng.uniform(0.85, 1.15)
    seq = np.clip(seq.astype("float32") * contrast + (brightness - 1) * 30, 0, 255).astype("uint8")

    return seq


def main():
    manifest = pd.read_csv("data/processed/manifest.csv")
    train_rows = manifest[manifest["split"] == "train"]
    new_rows = []

    for row in train_rows.itertuples():
        sequence = np.load(row.path)
        for i in range(N_AUGMENTED_COPIES):
            aug_seq = augment_sequence(sequence, seed=hash((row.video_id, i)) % (2**31))
            video_id = f"{row.video_id}_aug{i}"
            out_path = Path(row.path).parent / f"{row.clase}__{video_id}.npy"
            np.save(out_path, aug_seq)
            new_rows.append({
                "video_id": video_id, "clase": row.clase, "split": "train",
                "path": str(out_path), "origen": f"{row.origen}_augmented",
            })

    manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
    manifest.to_csv("data/processed/manifest.csv", index=False)
    print(f"Agregadas {len(new_rows)} secuencias aumentadas. "
          f"Train total ahora: {(manifest['split']=='train').sum()}")


if __name__ == "__main__":
    main()