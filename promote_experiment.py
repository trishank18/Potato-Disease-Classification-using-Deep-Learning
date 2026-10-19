"""Promote a variant selected using validation metrics to Flask/evaluate.py."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("custom", "weighted", "mobilenetv2"))
    args = parser.parse_args()
    experiment = ROOT / "models" / "experiments" / args.variant
    checkpoint = experiment / "best_val_macro_f1.keras"
    labels = experiment / "class_names.json"
    metrics = experiment / "validation_metrics.json"
    for path in (checkpoint, labels, metrics):
        if not path.is_file():
            raise SystemExit(f"Missing experiment artifact: {path}. Train this variant first.")
    summary = json.loads(metrics.read_text(encoding="utf-8"))
    if summary.get("class_names") != CLASS_NAMES or json.loads(labels.read_text(encoding="utf-8")) != CLASS_NAMES:
        raise SystemExit("Experiment class labels are inconsistent; refusing to promote.")
    model_dir = ROOT / "models"
    shutil.copy2(checkpoint, model_dir / "potato_cnn.keras")
    shutil.copy2(labels, model_dir / "class_names.json")
    (model_dir / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Promoted {args.variant} by its validation macro F1 ({summary['best_validation_macro_f1']:.4f}).")
    print("Now run evaluate.py once for final held-out test evaluation.")


if __name__ == "__main__":
    main()
