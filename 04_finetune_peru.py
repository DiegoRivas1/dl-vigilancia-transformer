"""
04_finetune_peru.py
Domain adaptation: parte de los pesos de un modelo ya entrenado en
UCF-Crime y continua el entrenamiento agregando la MITAD del dataset
peruano (50 Robbery + 50 Normal). La otra mitad queda intocada para
evaluacion final -- nunca se usa en fine-tuning.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import json
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from pathlib import Path
import argparse
from importlib import import_module

train_module = import_module("02_train")
build_feature_extractor = train_module.build_feature_extractor
build_model = train_module.build_model
extract_and_cache_features = train_module.extract_and_cache_features
plot_training_curves = train_module.plot_training_curves


def split_peru_finetune_holdout(peru_manifest: pd.DataFrame, classes: list, seed: int = 42):
    """Divide 50/50 por clase, de forma reproducible."""
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


def load_arrays(df: pd.DataFrame, classes: list):
    X = np.stack([np.load(p) for p in df["feature_path"]])
    y = np.array([classes.index(c) for c in df["clase"]])
    return X, y


def evaluate_holdout(model, holdout_df, classes, out_dir: Path):
    X_hold, y_hold = load_arrays(holdout_df, classes)
    y_pred = np.argmax(model.predict(X_hold), axis=1)

    true_labels_present = sorted(set(y_hold.tolist()))
    cm = confusion_matrix(y_hold, y_pred, labels=list(range(len(classes))))
    cm_display = cm[true_labels_present, :]
    row_labels = [classes[i] for i in true_labels_present]

    plt.figure(figsize=(8, 3.5))
    sns.heatmap(cm_display, annot=True, fmt="d", cmap="Greens",
                xticklabels=classes, yticklabels=row_labels)
    plt.xlabel("Prediccion del modelo")
    plt.ylabel("Clase real (Peru holdout, nunca visto)")
    plt.title("Post fine-tuning: UCF-Crime + 50%Peru -> Peru holdout")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_peru_holdout.png")
    plt.close()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_hold, y_pred, labels=true_labels_present, average=None,
    )
    metrics = {
        "accuracy_global": float(accuracy_score(y_hold, y_pred)),
        "por_clase": {
            classes[idx]: {"precision": float(p), "recall": float(r), "f1": float(f), "n_videos": int(s)}
            for idx, p, r, f, s in zip(true_labels_present, precision, recall, f1, support)
        },
    }
    with open(out_dir / "metrics_peru_holdout.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_experiment", default="exp01_baseline",
                         help="modelo ya entrenado del cual partir")
    parser.add_argument("--experiment", default="exp06_domain_adapted")
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--finetune_epochs", type=int, default=15)
    args = parser.parse_args()

    base_dir = Path("models") / args.base_experiment
    with open(base_dir / "config.json") as f:
        cfg = json.load(f)

    classes = cfg["classes"]
    seq_len = cfg["sequence_length"]
    img_size = cfg["img_size"]

    manifest = pd.read_csv("data/processed/manifest.csv")
    peru_manifest = manifest[manifest["split"] == "peru_eval"]
    if len(peru_manifest) == 0:
        raise RuntimeError("No hay secuencias peru_eval. Corre 01_preprocess.py primero.")

    feature_extractor = build_feature_extractor(img_size)
    peru_manifest = extract_and_cache_features(
        peru_manifest, feature_extractor, seq_len, img_size,
        cache_dir=Path("data/processed/features"),
    )

    peru_finetune, peru_holdout = split_peru_finetune_holdout(peru_manifest, classes)
    print(f"Peru dividido: {len(peru_finetune)} para fine-tuning, {len(peru_holdout)} para holdout final")

    ucf_manifest = manifest[manifest["split"].isin(["train", "val"])]
    ucf_manifest = extract_and_cache_features(
        ucf_manifest, feature_extractor, seq_len, img_size,
        cache_dir=Path("data/processed/features"),
    )
    ucf_train = ucf_manifest[ucf_manifest["split"] == "train"]
    ucf_val = ucf_manifest[ucf_manifest["split"] == "val"]

    # Train combinado: UCF-Crime train + mitad de Peru. Val se queda igual
    # que exp01 (solo UCF), para seleccionar checkpoint de forma consistente.
    combined_train = pd.concat([ucf_train, peru_finetune], ignore_index=True)
    X_train, y_train = load_arrays(combined_train, classes)
    X_val, y_val = load_arrays(ucf_val, classes)
    feat_dim = X_train.shape[-1]
    print(f"Train combinado: {X_train.shape} (UCF {len(ucf_train)} + Peru {len(peru_finetune)}), Val: {X_val.shape}")

    model = build_model(
        seq_len, feat_dim, num_classes=len(classes),
        dense_dim=cfg.get("dense_dim", 512),
        num_heads=cfg.get("num_heads", 4),
        dropout=cfg.get("dropout", 0.5),
    )
    model.load_weights(base_dir / "best_model.weights.h5")
    print(f"Pesos cargados desde {args.base_experiment}, continuando con LR={args.finetune_lr}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.finetune_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    exp_dir = Path("models") / args.experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg["finetune_from"] = args.base_experiment
    cfg["finetune_lr"] = args.finetune_lr
    cfg["finetune_epochs"] = args.finetune_epochs
    with open(exp_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(exp_dir / "best_model.weights.h5"),
            save_best_only=True, save_weights_only=True,
            monitor="val_accuracy", mode="max",
        ),
        keras.callbacks.CSVLogger(str(exp_dir / "history.csv")),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=cfg.get("batch_size", 32),
        epochs=args.finetune_epochs,
        callbacks=callbacks,
    )
    plot_training_curves(history.history, exp_dir / "training_curves.png")

    model.load_weights(exp_dir / "best_model.weights.h5")

    print("\n=== Evaluando contra Peru HOLDOUT (nunca visto ni en train ni en fine-tuning) ===")
    metrics = evaluate_holdout(model, peru_holdout, classes, exp_dir)
    print(f"Accuracy global: {metrics['accuracy_global']:.3f}")
    for clase, m in metrics["por_clase"].items():
        print(f"  {clase}: recall={m['recall']:.3f}  precision={m['precision']:.3f}  n={m['n_videos']}")


if __name__ == "__main__":
    main()