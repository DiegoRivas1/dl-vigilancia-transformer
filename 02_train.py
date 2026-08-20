"""
02_train.py
1) Extrae (y cachea) features por frame con un CNN preentrenado congelado.
2) Entrena un Transformer Encoder sobre esas secuencias de features.
3) Guarda checkpoints, curvas de entrenamiento, matriz de confusion y metricas.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import json
import yaml
import numpy as np
import pandas as pd
import keras
from keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from pathlib import Path
import argparse


# ---------------------------------------------------------------------------
# Config y utilidades
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_experiment_dir(name: str) -> Path:
    exp_dir = Path("models") / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


# ---------------------------------------------------------------------------
# 1) Feature extraction (CNN congelado) con cache en disco
# ---------------------------------------------------------------------------

def build_feature_extractor(img_size: int):
    base = keras.applications.EfficientNetB0(
        weights="imagenet", include_top=False, pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False
    inputs = keras.Input((img_size, img_size, 3))
    x = keras.applications.efficientnet.preprocess_input(inputs)
    outputs = base(x, training=False)
    return keras.Model(inputs, outputs, name="feature_extractor")


def extract_and_cache_features(manifest: pd.DataFrame, feature_extractor,
                                seq_len: int, img_size: int, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_paths = []

    for row in manifest.itertuples():
        cache_path = cache_dir / f"{row.split}__{row.clase}__{row.video_id}.npy"
        if not cache_path.exists():
            sequence = np.load(row.path)  # [seq_len, H, W, 3] uint8
            sequence = sequence.astype("float32")
            features = feature_extractor.predict(sequence, verbose=0)  # [seq_len, feat_dim]
            np.save(cache_path, features)
        feature_paths.append(str(cache_path))

    manifest = manifest.copy()
    manifest["feature_path"] = feature_paths
    return manifest


# ---------------------------------------------------------------------------
# 2) Dataset -> arrays en memoria (250 videos cabe sin problema)
# ---------------------------------------------------------------------------

def load_split_arrays(manifest: pd.DataFrame, split: str, classes: list):
    subset = manifest[manifest["split"] == split]
    X = np.stack([np.load(p) for p in subset["feature_path"]])
    y = np.array([classes.index(c) for c in subset["clase"]])
    return X, y


# ---------------------------------------------------------------------------
# 3) Modelo: Positional Embedding + Transformer Encoder (igual al ejemplo Keras)
# ---------------------------------------------------------------------------

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        length = keras.ops.shape(inputs)[1]
        positions = keras.ops.arange(0, length, 1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.3)
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, training=False):
        attn_out = self.attention(inputs, inputs, training=training)
        x = self.layernorm_1(inputs + attn_out)
        proj_out = self.dense_proj(x)
        return self.layernorm_2(x + proj_out)


def build_model(seq_len, feat_dim, num_classes, embed_dim=None, dense_dim=512, num_heads=4, dropout=0.5):
    embed_dim = embed_dim or feat_dim
    inputs = keras.Input((seq_len, feat_dim))
    x = PositionalEmbedding(seq_len, feat_dim)(inputs)
    x = TransformerEncoder(feat_dim, dense_dim, num_heads)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="video_transformer_classifier")


# ---------------------------------------------------------------------------
# 4) Entrenamiento + guardado de artefactos por experimento
# ---------------------------------------------------------------------------

def plot_training_curves(history, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, classes, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Prediccion"); plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", required=True, help="nombre del experimento, ej: exp01_baseline")
    parser.add_argument("--resume", action="store_true", help="continuar desde last_checkpoint.weights.h5")
    args = parser.parse_args()

    cfg = load_config(args.config)
    classes = cfg["classes"]
    seq_len = cfg["sequence_length"]
    img_size = cfg["img_size"]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    lr = cfg["learning_rate"]

    exp_dir = get_experiment_dir(args.experiment)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    manifest = pd.read_csv("data/processed/manifest.csv")
    manifest = manifest[manifest["split"].isin(["train", "val"])]  # peru_eval queda fuera del entrenamiento

    print("Extrayendo/cargando features cacheadas...")
    feature_extractor = build_feature_extractor(img_size)
    manifest = extract_and_cache_features(
        manifest, feature_extractor, seq_len, img_size,
        cache_dir=Path("data/processed/features"),
    )

    X_train, y_train = load_split_arrays(manifest, "train", classes)
    X_val, y_val = load_split_arrays(manifest, "val", classes)
    feat_dim = X_train.shape[-1]
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, feat_dim={feat_dim}")

    model = build_model(
        seq_len, feat_dim, num_classes=len(classes),
        dense_dim=cfg.get("dense_dim", 512),
        num_heads=cfg.get("num_heads", 4),
        dropout=cfg.get("dropout", 0.5),
    )

    last_ckpt = exp_dir / "last_checkpoint.weights.h5"
    if args.resume and last_ckpt.exists():
        model.load_weights(last_ckpt)
        print(f"Reanudando desde {last_ckpt}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(exp_dir / "best_model.weights.h5"),
            save_best_only=True, save_weights_only=True,
            monitor="val_accuracy", mode="max",
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(last_ckpt), save_weights_only=True,
        ),
        keras.callbacks.CSVLogger(str(exp_dir / "history.csv"), append=args.resume),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
    )

    plot_training_curves(history.history, exp_dir / "training_curves.png")

    model.load_weights(exp_dir / "best_model.weights.h5")
    y_pred = np.argmax(model.predict(X_val), axis=1)

    plot_confusion_matrix(y_val, y_pred, classes, exp_dir / "confusion_matrix.png")

    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="macro")
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nMetricas finales (val):", metrics)


if __name__ == "__main__":
    main()