"""
01_preprocess.py
Extrae frames de los videos crudos y arma secuencias de longitud fija
por video, usando el split oficial train_001/test_001 del paper.

Estructura esperada de entrada:
    data/raw/ucf_crime/<Clase>/video.mp4          (Robbery, Burglary, Shoplifting, Stealing, Normal)
    data/raw/UCF_Crimes-Train-Test-Split/Action_Regnition_splits/train_001.txt
    data/raw/UCF_Crimes-Train-Test-Split/Action_Regnition_splits/test_001.txt
    data/raw/peru_robos/<Clase>/...                (opcional, para evaluación final)

Salida:
    data/processed/<split>/<clase>__<video_id>.npy   (array [seq_len, H, W, 3] uint8)
    data/processed/manifest.csv
"""

import cv2
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm
import argparse

TARGET_CLASSES = ["Robbery", "Burglary", "Shoplifting", "Stealing"]
NORMAL_CLASS_IN_SPLIT = "Normal_Videos_event"
SPLIT_DIR = Path("data/raw/UCF_Crimes-Train-Test-Split/Action_Regnition_splits")


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_official_split(fold: str = "001") -> dict:
    """Devuelve {'Robbery048_x264.mp4': 'train'|'val'} usando el split oficial."""
    split_map = {}
    for split_name, tag in (("train", "train"), ("test", "val")):
        f = SPLIT_DIR / f"{split_name}_{fold}.txt"
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            filename = line.split("/")[-1]
            split_map[filename] = tag
    return split_map


def sample_frame_indices(total_frames: int, seq_len: int) -> np.ndarray:
    """Muestreo uniforme de seq_len indices a lo largo de todo el video.
    Si el video tiene menos frames que seq_len, repite el ultimo frame
    (padding) en vez de descartar el video."""
    if total_frames <= 0:
        return np.zeros(seq_len, dtype=int)
    if total_frames >= seq_len:
        return np.linspace(0, total_frames - 1, seq_len).astype(int)
    indices = np.arange(total_frames)
    pad = np.full(seq_len - total_frames, total_frames - 1)
    return np.concatenate([indices, pad])


def extract_sequence(video_path: Path, seq_len: int, img_size: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] No se pudo abrir: {video_path.name}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = set(sample_frame_indices(total_frames, seq_len).tolist())

    frames_by_index = {}
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in target_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frames_by_index[idx] = frame
        idx += 1
    cap.release()

    if not frames_by_index:
        return None

    ordered_indices = sample_frame_indices(total_frames, seq_len)
    sequence = []
    last_valid = next(iter(frames_by_index.values()))
    for i in ordered_indices:
        frame = frames_by_index.get(int(i), last_valid)
        last_valid = frame
        sequence.append(frame)

    return np.stack(sequence, axis=0).astype(np.uint8)


def process_ucf_crime(raw_dir: Path, out_dir: Path, classes: list,
                       seq_len: int, img_size: int, split_map: dict,
                       manifest_rows: list):
    for clase in classes:
        class_dir = raw_dir / clase
        if not class_dir.exists():
            print(f"  [SKIP] No existe carpeta para clase '{clase}' en {raw_dir}")
            continue

        videos = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))
        print(f"\nClase '{clase}': {len(videos)} videos encontrados")

        for video_path in tqdm(videos, desc=f"ucf/{clase}"):
            split_name = split_map.get(video_path.name)
            if split_name is None:
                split_name = "train"  # video adicional, fuera del split oficial -> siempre a train
                print(f"  [INFO] {video_path.name} no esta en el split oficial UCF-Crime, "
                      f"se asigna a train (fuente adicional)")

            sequence = extract_sequence(video_path, seq_len, img_size)
            if sequence is None:
                continue

            video_id = video_path.stem
            out_name = f"{clase}__{video_id}.npy"
            out_path = out_dir / split_name / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, sequence)

            manifest_rows.append({
                "video_id": video_id,
                "clase": clase,
                "split": split_name,
                "path": str(out_path),
                "origen": "ucf_crime",
            })


def process_peru(raw_dir: Path, out_dir: Path, classes: list,
                  seq_len: int, img_size: int, manifest_rows: list):
    if not raw_dir.exists():
        print(f"\n[INFO] {raw_dir} no existe todavia, se omite dataset peruano por ahora")
        return

    for clase in classes:
        class_dir = raw_dir / clase
        if not class_dir.exists():
            continue
        videos = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))
        print(f"\n[PERU] Clase '{clase}': {len(videos)} videos")

        for video_path in tqdm(videos, desc=f"peru/{clase}"):
            sequence = extract_sequence(video_path, seq_len, img_size)
            if sequence is None:
                continue

            video_id = video_path.stem
            out_name = f"{clase}__{video_id}.npy"
            out_path = out_dir / "peru_eval" / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, sequence)

            manifest_rows.append({
                "video_id": video_id,
                "clase": clase,
                "split": "peru_eval",
                "path": str(out_path),
                "origen": "peru_robos",
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fold", default="001", help="fold del split oficial (001-004)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    classes = cfg["classes"]  # ["Robbery", "Burglary", "Shoplifting", "Stealing", "Normal"]
    seq_len = cfg["sequence_length"]
    img_size = cfg["img_size"]

    processed_dir = Path("data/processed")
    manifest_rows = []

    split_map = load_official_split(fold=args.fold)
    print(f"Split oficial cargado: {len(split_map)} videos mapeados (fold {args.fold})")

    process_ucf_crime(
        raw_dir=Path("data/raw/ucf_crime"),
        out_dir=processed_dir,
        classes=classes,
        seq_len=seq_len,
        img_size=img_size,
        split_map=split_map,
        manifest_rows=manifest_rows,
    )

    process_peru(
        raw_dir=Path("data/raw/peru_robos"),
        out_dir=processed_dir,
        classes=classes,
        seq_len=seq_len,
        img_size=img_size,
        manifest_rows=manifest_rows,
    )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = processed_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifest guardado en {manifest_path} ({len(manifest)} secuencias)")
    print(manifest.groupby(["split", "clase"]).size())


if __name__ == "__main__":
    main()