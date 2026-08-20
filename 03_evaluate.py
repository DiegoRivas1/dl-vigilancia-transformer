"""
03_evaluate.py
Evalua un modelo ya entrenado contra el split peru_eval (dataset peruano),
nunca visto durante entrenamiento. Reconstruye la arquitectura desde el
config.json guardado por ese experimento, para evitar desajustes.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import json
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, classification_report
from pathlib import Path
import argparse

# Reusa las mismas clases/capas custom que 02_train.py -- deben ser
# identicas para que load_weights funcione correctamente.
from importlib import import_module
train_module = import_module("02_train")
build_feature_extractor = train_module.build_feature_extractor
build_model = train_module.build_model
extract_and_cache_features = train_module.extract_and_cache_features


def load_split_arrays(manifest: pd.DataFrame, split: str, classes: list):
    subset = manifest[manifest["split"] == split]
    X = np.stack([np.load(p) for p in subset["feature_path"]])
    y = np.array([classes.index(c) for c in subset["clase"]])
    video_ids = subset["video_id"].tolist()
    return X, y, video_ids


def plot_confusion_matrix(y_true, y_pred, classes, out_path: Path, true_labels_present):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    # Solo mostramos filas de clases que realmente existen en el dataset peruano
    cm_display = cm[true_labels_present, :]
    row_labels = [classes[i] for i in true_labels_present]

    plt.figure(figsize=(8, 3.5))
    sns.heatmap(cm_display, annot=True, fmt="d", cmap="Oranges",
                xticklabels=classes, yticklabels=row_labels)
    plt.xlabel("Prediccion del modelo")
    plt.ylabel("Clase real (Peru)")
    plt.title("Generalizacion cross-domain: UCF-Crime -> Peru")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return cm


def per_class_breakdown(y_true, y_pred, classes, true_labels_present):
    """Para cada clase real presente en Peru, que porcentaje cayo en cada
    clase predicha -- util para ver hacia donde se van los errores."""
    rows = []
    for true_idx in true_labels_present:
        mask = y_true == true_idx
        preds_for_class = y_pred[mask]
        total = len(preds_for_class)
        for pred_idx, pred_clase in enumerate(classes):
            count = int(np.sum(preds_for_class == pred_idx))
            if count > 0:
                rows.append({
                    "clase_real_peru": classes[true_idx],
                    "predicho_como": pred_clase,
                    "cantidad": count,
                    "porcentaje": round(100 * count / total, 1),
                })
    return pd.DataFrame(rows).sort_values(["clase_real_peru", "cantidad"], ascending=[True, False])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="exp01_baseline",
                         help="carpeta en models/ del experimento a evaluar")
    args = parser.parse_args()

    exp_dir = Path("models") / args.experiment
    with open(exp_dir / "config.json") as f:
        cfg = json.load(f)

    classes = cfg["classes"]
    seq_len = cfg["sequence_length"]
    img_size = cfg["img_size"]

    manifest = pd.read_csv("data/processed/manifest.csv")
    peru_manifest = manifest[manifest["split"] == "peru_eval"]

    if len(peru_manifest) == 0:
        raise RuntimeError("No hay secuencias 'peru_eval' en el manifest. "
                            "Corre 01_preprocess.py con data/raw/peru_robos/ poblado.")

    print(f"Evaluando experimento '{args.experiment}' contra {len(peru_manifest)} videos peruanos")

    feature_extractor = build_feature_extractor(img_size)
    peru_manifest = extract_and_cache_features(
        peru_manifest, feature_extractor, seq_len, img_size,
        cache_dir=Path("data/processed/features"),
    )

    X_peru, y_peru, video_ids = load_split_arrays(peru_manifest, "peru_eval", classes)
    feat_dim = X_peru.shape[-1]
    print(f"Peru eval: {X_peru.shape}")

    model = build_model(
        seq_len, feat_dim, num_classes=len(classes),
        dense_dim=cfg.get("dense_dim", 512),
        num_heads=cfg.get("num_heads", 4),
        dropout=cfg.get("dropout", 0.5),
    )
    model.load_weights(exp_dir / "best_model.weights.h5")

    y_pred_probs = model.predict(X_peru)
    y_pred = np.argmax(y_pred_probs, axis=1)

    true_labels_present = sorted(set(y_peru.tolist()))

    out_dir = exp_dir / "peru_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = plot_confusion_matrix(y_peru, y_pred, classes,
                                out_dir / "confusion_matrix_peru.png",
                                true_labels_present)

    breakdown = per_class_breakdown(y_peru, y_pred, classes, true_labels_present)
    breakdown.to_csv(out_dir / "breakdown_peru.csv", index=False)

    # Metricas solo tienen sentido calculadas sobre las clases que Peru
    # realmente contiene (Robbery, Normal) -- no reportamos "accuracy" de
    # 5 clases porque Peru nunca ofrece 3 de ellas como verdad.
    mask_present = np.isin(y_peru, true_labels_present)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_peru[mask_present], y_pred[mask_present],
        labels=true_labels_present, average=None,
    )

    metrics = {
        "accuracy_global": float(accuracy_score(y_peru, y_pred)),
        "por_clase": {
            classes[idx]: {
                "precision": float(p), "recall": float(r),
                "f1": float(f), "n_videos": int(s),
            }
            for idx, p, r, f, s in zip(true_labels_present, precision, recall, f1, support)
        },
    }

    with open(out_dir / "metrics_peru.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n=== Resultados sobre dataset peruano ===")
    print(f"Accuracy global (contando confusion con Burglary/Shoplifting/Stealing como error): "
          f"{metrics['accuracy_global']:.3f}")
    for clase, m in metrics["por_clase"].items():
        print(f"  {clase}: recall={m['recall']:.3f}  precision={m['precision']:.3f}  n={m['n_videos']}")
    print(f"\nDetalle de confusiones guardado en {out_dir / 'breakdown_peru.csv'}")
    print(f"Matriz de confusion guardada en {out_dir / 'confusion_matrix_peru.png'}")


if __name__ == "__main__":
    main()