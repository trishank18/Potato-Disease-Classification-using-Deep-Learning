"""Evaluate the promoted checkpoint once on the held-out test split."""
from collections import Counter
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support, ConfusionMatrixDisplay)

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


def main() -> None:
    model_path = ROOT / "models" / "potato_cnn.keras"
    labels_path = ROOT / "models" / "class_names.json"
    test_path = ROOT / "data" / "processed" / "test"
    for path in (model_path, labels_path, test_path):
        if not path.exists():
            raise SystemExit(f"Required file/folder missing: {path}. Run split_dataset.py and train.py first.")
    class_names = json.loads(labels_path.read_text(encoding="utf-8"))
    if class_names != CLASS_NAMES:
        raise SystemExit(f"Saved class order {class_names} differs from required order {CLASS_NAMES}")
    found = sorted(p.name for p in test_path.iterdir() if p.is_dir())
    if found != sorted(class_names):
        raise SystemExit(f"Test folders do not match saved labels: {found}")
    actual_counts = {name: sum(1 for p in (test_path / name).iterdir() if p.is_file()) for name in class_names}
    ds = tf.keras.utils.image_dataset_from_directory(
        test_path, labels="inferred", label_mode="int", class_names=class_names,
        color_mode="rgb", image_size=IMAGE_SIZE, interpolation="bilinear",
        batch_size=BATCH_SIZE, shuffle=False)
    model = tf.keras.models.load_model(model_path)
    if tuple(model.input_shape[1:]) != (*IMAGE_SIZE, 3) or model.output_shape[-1] != len(class_names):
        raise SystemExit(f"Model shape {model.input_shape} -> {model.output_shape} does not match expected 128x128 RGB and 3 classes")
    test_loss, test_accuracy = model.evaluate(ds, verbose=1)
    y_true = np.concatenate([y.numpy().reshape(-1) for _, y in ds])
    probabilities = model.predict(ds, verbose=1)
    y_pred = probabilities.argmax(axis=1)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    text_report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                                average="weighted", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    predicted_counts = Counter(class_names[int(i)] for i in y_pred)
    print("Ordered classes:", list(enumerate(class_names)))
    print("Actual class distribution:", actual_counts)
    print("Predicted class distribution:", {name: predicted_counts.get(name, 0) for name in class_names})
    print(f"Test loss: {test_loss:.4f}\nTest accuracy: {test_accuracy:.4f}")
    print(f"Weighted precision: {precision:.4f}\nWeighted recall: {recall:.4f}\nWeighted F1: {f1:.4f}\nMacro F1: {macro_f1:.4f}")
    print("\nPer-class precision / recall / F1 and correct / incorrect counts:")
    for i, name in enumerate(class_names):
        support = int(cm[i].sum()); correct = int(cm[i, i]); incorrect = support - correct
        row = report[name]
        print(f"  {name}: precision={row['precision']:.4f}, recall={row['recall']:.4f}, f1={row['f1-score']:.4f}, support={support}, correct={correct}, incorrect={incorrect}")
    print("\nClassification report:\n" + text_report)
    if len(set(y_pred.tolist())) == 1:
        predicted = class_names[int(y_pred[0])]
        print(f"WARNING: model predicted only {predicted}. Check class-to-index mapping, data labels and image quality; compare train/val distributions and confirm preprocessing is identical. Class weights do not fix mislabeled or domain-shifted data.")

    out = ROOT / "outputs"; out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Greens", xticks_rotation=30, colorbar=False)
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=160); plt.close(fig)
    (out / "classification_report.txt").write_text(
        f"Test loss: {test_loss:.6f}\nTest accuracy: {test_accuracy:.6f}\nMacro F1: {macro_f1:.6f}\nWeighted precision: {precision:.6f}\nWeighted recall: {recall:.6f}\nWeighted F1: {f1:.6f}\n\nActual distribution: {actual_counts}\nPredicted distribution: {dict(predicted_counts)}\n\n{text_report}", encoding="utf-8")
    with (out / "per_class_test_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "actual", "predicted", "correct", "incorrect", "precision", "recall", "f1"])
        writer.writeheader()
        for i, name in enumerate(class_names):
            writer.writerow({"class": name, "actual": int(cm[i].sum()), "predicted": int(cm[:, i].sum()),
                             "correct": int(cm[i, i]), "incorrect": int(cm[i].sum() - cm[i, i]),
                             "precision": report[name]["precision"], "recall": report[name]["recall"], "f1": report[name]["f1-score"]})
    print(f"Saved outputs/confusion_matrix.png, outputs/classification_report.txt, and outputs/per_class_test_counts.csv")


if __name__ == "__main__":
    main()
