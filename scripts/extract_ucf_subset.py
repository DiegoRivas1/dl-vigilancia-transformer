"""
scripts/extract_ucf_subset.py  (v3 - matching por nombre de archivo)
"""

import zipfile
from pathlib import Path

TARGET_CLASSES = ["Robbery", "Burglary", "Shoplifting", "Stealing"]
NORMAL_CLASS_IN_SPLIT = "Normal_Videos_event"
SPLIT_DIR = Path("data/raw/UCF_Crimes-Train-Test-Split/Action_Regnition_splits")
OUT_DIR = Path("data/raw/ucf_crime")


def load_target_files(fold: str = "001") -> dict:
    """Devuelve {'Robbery048_x264.mp4': 'Robbery', ...} -- clave = solo el nombre de archivo."""
    targets = {}
    for split_name in ("train", "test"):
        f = SPLIT_DIR / f"{split_name}_{fold}.txt"
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            clase_original = line.split("/")[0]
            filename = line.split("/")[-1]
            if clase_original in TARGET_CLASSES:
                targets[filename] = clase_original
            elif clase_original == NORMAL_CLASS_IN_SPLIT:
                targets[filename] = "Normal"
    return targets


def inspect_and_extract(zip_path: Path, targets: dict):
    with zipfile.ZipFile(zip_path) as zf:
        namelist = zf.namelist()
        extracted = 0
        for member in namelist:
            if member.endswith("/"):
                continue
            filename = Path(member).name
            if filename not in targets:
                continue
            clase = targets[filename]
            dest_dir = OUT_DIR / clase
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / filename
            if dest_path.exists():
                continue
            with zf.open(member) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())
            extracted += 1
        print(f"{zip_path.name} -> {extracted} videos nuevos extraidos")


if __name__ == "__main__":
    targets = load_target_files(fold="001")
    print(f"Buscamos {len(targets)} videos en total")

    zips_to_check = list(Path("data/raw/downloads").glob("*.zip"))
    for zp in zips_to_check:
        inspect_and_extract(zp, targets)