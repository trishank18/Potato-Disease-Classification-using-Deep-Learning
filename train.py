"""Train a controlled CNN experiment using train/ and val/ only.

Examples:
  python train.py --variant custom       # unweighted baseline
  python train.py --variant weighted      # custom CNN + inverse-frequency weights
  python train.py --variant mobilenetv2   # selected default; downloads weights if needed
"""
from collections import Counter
import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import f1_score, recall_score

ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = (128, 128)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
SEED = 42
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
AUTOTUNE = tf.data.AUTOTUNE


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def count_images(path: Path) -> dict[str, int]:
    counts = {}
    for name in CLASS_NAMES:
        folder = path / name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class directory: {folder}. Run split_dataset.py and dataset_verify.py.")
        counts[name] = sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"})
        if counts[name] == 0:
            raise ValueError(f"No images in {folder}")
    return counts


def load_split(path: Path, shuffle: bool):
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {path}. Run python split_dataset.py first.")
    found = sorted(p.name for p in path.iterdir() if p.is_dir())
    if found != sorted(CLASS_NAMES):
        raise ValueError(f"Expected class folders {CLASS_NAMES} in {path}; found {found}")
    # Keras decodes RGB and resizes to 128x128. The dataset remains float32 in 0..255;
    # normalization happens exactly once inside the saved model, in train/eval/Flask alike.
    ds = tf.keras.utils.image_dataset_from_directory(
        path, labels="inferred", label_mode="int", class_names=CLASS_NAMES,
        color_mode="rgb", image_size=IMAGE_SIZE, interpolation="bilinear",
        batch_size=None, shuffle=shuffle, seed=SEED)
    return ds.batch(DEFAULT_BATCH_SIZE).prefetch(AUTOTUNE)


class ValidationMetrics(tf.keras.callbacks.Callback):
    """Compute validation macro F1 and per-class recall for checkpoint selection."""
    def __init__(self, validation_data, num_classes: int):
        super().__init__()
        self.validation_data = validation_data
        self.num_classes = num_classes
        self.y_true = np.concatenate([y.numpy().reshape(-1) for _, y in validation_data])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        prob = self.model.predict(self.validation_data, verbose=0)
        pred = prob.argmax(axis=1)
        logs["val_macro_f1"] = float(f1_score(self.y_true, pred, labels=range(self.num_classes), average="macro", zero_division=0))
        recalls = recall_score(self.y_true, pred, labels=range(self.num_classes), average=None, zero_division=0)
        for index, value in enumerate(recalls):
            logs[f"val_recall_class_{index}"] = float(value)
        print("  val_macro_f1: {:.4f} | class recalls: {}".format(logs["val_macro_f1"], np.round(recalls, 3).tolist()))


def augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
    ], name="training_augmentation")


def build_model(variant: str, num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input((*IMAGE_SIZE, 3), name="leaf_rgb_0_255")
    x = augmentation_layer()(inputs)
    if variant == "mobilenetv2":
        # MobileNetV2 preprocessing maps RGB values 0..255 to [-1, 1].
        x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1, name="mobilenetv2_preprocessing")(x)
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3), pooling="avg")
        backbone.trainable = False
        x = backbone(x, training=False)
        x = tf.keras.layers.Dropout(0.35)(x)
    else:
        x = tf.keras.layers.Rescaling(1.0 / 255.0, name="pixel_normalization")(x)
        regularizer = tf.keras.regularizers.l2(1e-4) if variant == "weighted" else None
        for filters in (32, 64, 128, 256):
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False,
                kernel_regularizer=regularizer)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer)(x)
        x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name=f"potato_leaf_{variant}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def make_class_weights(counts: dict[str, int]) -> dict[int, float]:
    total = sum(counts.values())
    n_classes = len(CLASS_NAMES)
    return {i: total / (n_classes * counts[name]) for i, name in enumerate(CLASS_NAMES)}


def save_curves(history, out_dir: Path, variant: str) -> None:
    for key, label in (("accuracy", "Accuracy"), ("loss", "Loss")):
        plt.figure(figsize=(7, 4))
        plt.plot(history.history.get(key, []), label="Training")
        plt.plot(history.history.get(f"val_{key}", []), label="Validation")
        plt.xlabel("Epoch"); plt.ylabel(label); plt.title(f"{variant}: training and validation {label.lower()}")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"{variant}_{key}_curve.png", dpi=150); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("custom", "weighted", "mobilenetv2"), default="mobilenetv2")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--no-promote", action="store_true", help="Keep experiment checkpoint without making it the Flask model")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    set_seeds()
    train_path = ROOT / "data" / "processed" / "train"
    val_path = ROOT / "data" / "processed" / "val"
    train_counts, val_counts = count_images(train_path), count_images(val_path)
    # Decode/cache once, then shuffle already-decoded examples in memory. This avoids
    # repeated reads from cloud-synced (e.g. OneDrive Files On-Demand) dataset paths.
    def dataset(path, shuffle):
        raw = tf.keras.utils.image_dataset_from_directory(
            path, labels="inferred", label_mode="int", class_names=CLASS_NAMES,
            color_mode="rgb", image_size=IMAGE_SIZE, interpolation="bilinear",
            batch_size=None, shuffle=False)
        raw = raw.cache()
        if shuffle:
            raw = raw.shuffle(buffer_size=sum(count_images(path).values()), seed=SEED,
                              reshuffle_each_iteration=True)
        raw = raw.batch(args.batch_size)
        return raw.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    train_ds, val_ds = dataset(train_path, True), dataset(val_path, False)
    print(f"Variant: {args.variant}; seed: {SEED}; image size: {IMAGE_SIZE}; batch: {args.batch_size}; max epochs: {args.epochs}")
    print("Class order and integer labels:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name} | train={train_counts[name]} | val={val_counts[name]}")
    print(f"Class imbalance ratio (largest/smallest training class): {max(train_counts.values()) / min(train_counts.values()):.2f}:1")
    class_weights = make_class_weights(train_counts) if args.variant in {"weighted", "mobilenetv2"} else None
    if class_weights:
        print("Balanced class weights:", {CLASS_NAMES[i]: round(w, 3) for i, w in class_weights.items()})
    else:
        print("Class weighting: disabled (unweighted baseline)")

    model = build_model(args.variant, len(CLASS_NAMES))
    model.summary()
    model_dir = ROOT / "models"
    out_dir = ROOT / "outputs"
    experiment_dir = model_dir / "experiments" / args.variant
    experiment_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = experiment_dir / "class_names.json"
    labels_path.write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")
    val_loss_path = experiment_dir / "best_val_loss.keras"
    macro_path = experiment_dir / "best_val_macro_f1.keras"
    metric_callback = ValidationMetrics(val_ds, len(CLASS_NAMES))
    callbacks = [
        metric_callback,
        tf.keras.callbacks.ModelCheckpoint(val_loss_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(macro_path, monitor="val_macro_f1", mode="max", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=8, restore_best_weights=False, verbose=1),
        tf.keras.callbacks.CSVLogger(experiment_dir / "training_log.csv"),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=callbacks, class_weight=class_weights)
    # The deployed candidate is selected by validation macro F1; also preserve best val-loss checkpoint.
    best = tf.keras.models.load_model(macro_path)
    model_dir.mkdir(exist_ok=True)
    summary = {
        "variant": args.variant,
        "checkpoint_selected_by": "validation_macro_f1",
        "best_validation_macro_f1": float(max(history.history["val_macro_f1"])),
        "best_validation_loss": float(min(history.history["val_loss"])),
        "epochs_completed": len(history.history["loss"]),
        "class_weights": class_weights,
        "class_names": CLASS_NAMES,
        "train_counts": train_counts,
        "validation_counts": val_counts,
        "seed": SEED,
    }
    (experiment_dir / "validation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_promote:
        shutil.copy2(macro_path, model_dir / "potato_cnn.keras")
        shutil.copy2(labels_path, model_dir / "class_names.json")
        (model_dir / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_curves(history, out_dir, args.variant)
    print(f"Best observed validation macro F1: {max(history.history['val_macro_f1']):.4f}")
    print(f"Best observed validation loss: {min(history.history['val_loss']):.4f}")
    print(f"Saved checkpoints: {val_loss_path} (lowest val loss), {macro_path} (highest val macro F1)")
    if not args.no_promote:
        print("Promoted best validation macro-F1 checkpoint to models/potato_cnn.keras for Flask/evaluate.py")
    print("Test data was not loaded or used.")


if __name__ == "__main__":
    main()
