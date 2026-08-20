"""
05_live_demo.py
Deteccion en vivo (webcam o archivo de video) usando un modelo ya
entrenado. Mantiene un buffer circular de los ultimos sequence_length
frames y corre inferencia periodicamente, dibujando la prediccion
sobre el video.

Uso:
    python 05_live_demo.py --experiment exp06_domain_adapted --source 0
    python 05_live_demo.py --experiment exp06_domain_adapted --source ruta/video.mp4 --save salida.mp4
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import json
import time
import argparse
import numpy as np
import cv2
import keras
from collections import deque
from pathlib import Path
from importlib import import_module

train_module = import_module("02_train")
build_feature_extractor = train_module.build_feature_extractor
build_model = train_module.build_model

# Colores BGR (OpenCV) por clase -- rojo para eventos sospechosos, verde para Normal
CLASS_COLORS = {
    "Robbery": (0, 0, 255),
    "Burglary": (0, 69, 255),
    "Shoplifting": (0, 140, 255),
    "Stealing": (0, 165, 255),
    "Normal": (0, 200, 0),
}

INFER_EVERY_N_FRAMES = 5   # recalcular prediccion cada N frames capturados
CONFIDENCE_THRESHOLD = 0.4  # por debajo de esto, se muestra "Analizando..."


def load_experiment(experiment: str):
    exp_dir = Path("models") / experiment
    with open(exp_dir / "config.json") as f:
        cfg = json.load(f)

    feature_extractor = build_feature_extractor(cfg["img_size"])
    feat_dim = feature_extractor.output_shape[-1]

    model = build_model(
        cfg["sequence_length"], feat_dim, num_classes=len(cfg["classes"]),
        dense_dim=cfg.get("dense_dim", 512),
        num_heads=cfg.get("num_heads", 4),
        dropout=cfg.get("dropout", 0.5),
    )
    model.load_weights(exp_dir / "best_model.weights.h5")

    return feature_extractor, model, cfg


def predict_on_buffer(frame_buffer: deque, feature_extractor, model, img_size: int, classes: list):
    sequence = np.stack(list(frame_buffer), axis=0).astype("float32")  # [seq_len, img_size, img_size, 3]
    features = feature_extractor.predict(sequence, verbose=0)          # [seq_len, feat_dim]
    features = np.expand_dims(features, axis=0)                        # [1, seq_len, feat_dim]
    probs = model.predict(features, verbose=0)[0]                      # [num_classes]
    pred_idx = int(np.argmax(probs))
    return classes[pred_idx], float(probs[pred_idx]), probs


def draw_overlay(frame, label, confidence, fps, buffer_ready: bool):
    h, w = frame.shape[:2]
    color = CLASS_COLORS.get(label, (200, 200, 200))

    if not buffer_ready:
        text = "Analizando..."
        color = (150, 150, 150)
    elif confidence < CONFIDENCE_THRESHOLD:
        text = f"Incierto ({label} {confidence*100:.0f}%)"
        color = (150, 150, 150)
    else:
        text = f"{label}: {confidence*100:.0f}%"

    # Franja superior semitransparente para el texto
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, text, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Borde de color alrededor del frame como alerta visual
    if buffer_ready and confidence >= CONFIDENCE_THRESHOLD and label != "Normal":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="exp06_domain_adapted")
    parser.add_argument("--source", default="0", help="'0' para webcam, o ruta a un archivo de video")
    parser.add_argument("--save", default=None, help="ruta de salida .mp4 para grabar la demo")
    args = parser.parse_args()

    feature_extractor, model, cfg = load_experiment(args.experiment)
    classes = cfg["classes"]
    seq_len = cfg["sequence_length"]
    img_size = cfg["img_size"]

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {args.source}")

    frame_buffer = deque(maxlen=seq_len)
    current_label, current_conf = "Normal", 0.0
    frame_count = 0
    prev_time = time.time()

    writer = None
    if args.save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 20.0, (w, h))
        print(f"Grabando salida en {args.save}")

    print("Presiona 'q' para salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        small = cv2.resize(frame, (img_size, img_size))
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        frame_buffer.append(small_rgb)

        buffer_ready = len(frame_buffer) == seq_len
        if buffer_ready and frame_count % INFER_EVERY_N_FRAMES == 0:
            current_label, current_conf, _ = predict_on_buffer(
                frame_buffer, feature_extractor, model, img_size, classes
            )

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        frame = draw_overlay(frame, current_label, current_conf, fps, buffer_ready)

        cv2.imshow("Deteccion de eventos sospechosos - dl-vigilancia-transformer", frame)
        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()