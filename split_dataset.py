"""Create a reproducible, class-stratified 60/20/20 split without touching raw data."""
from collections import Counter
from pathlib import Path
import random
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CLASSES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
SPLITS = ("train", "test", "val")
SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def counts_for(n: int) -> tuple[int, int, int]:
    """Largest-remainder allocation; split counts sum exactly to n."""
    raw = [0.60 * n, 0.20 * n, 0.20 * n]
    sizes = [int(x) for x in raw]
    order = sorted(range(3), key=lambda i: (raw[i] - sizes[i], -i), reverse=True)
    for i in order[:n - sum(sizes)]:
        sizes[i] += 1
    return tuple(sizes)


def main() -> None:
    missing = [name for name in CLASSES if not (RAW / name).is_dir()]
    if missing:
        raise SystemExit(f"Missing required raw class folders under {RAW}: {', '.join(missing)}")
    extra = sorted(p.name for p in RAW.iterdir() if p.is_dir() and p.name not in CLASSES)
    if extra:
        raise SystemExit(f"Unexpected folders under {RAW}: {extra}. Move non-potato classes elsewhere.")
    files_by_class = {}
    for name in CLASSES:
        files = sorted(p for p in (RAW / name).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise SystemExit(f"No supported images found in {RAW / name}")
        files_by_class[name] = files

    # Build the replacement away from the current processed tree. Raw inputs are never modified.
    with tempfile.TemporaryDirectory(prefix="potato_processed_", dir=ROOT / "data") as temp:
        stage = Path(temp) / "processed"
        for split in SPLITS:
            for name in CLASSES:
                (stage / split / name).mkdir(parents=True, exist_ok=True)
        rng = random.Random(SEED)
        totals = Counter()
        print(f"{'Class':24} {'Total':>6} {'Train':>7} {'Test':>6} {'Val':>6}")
        for name in CLASSES:
            files = files_by_class[name][:]
            rng.shuffle(files)
            n_train, n_test, n_val = counts_for(len(files))
            groups = {
                "train": files[:n_train],
                "test": files[n_train:n_train + n_test],
                "val": files[n_train + n_test:],
            }
            for split, selected in groups.items():
                totals[split] += len(selected)
                for src in selected:
                    shutil.copy2(src, stage / split / name / src.name)
            print(f"{name:24} {len(files):6} {n_train:7} {n_test:6} {n_val:6}")
        print(f"{'TOTAL':24} {sum(totals.values()):6} {totals['train']:7} {totals['test']:6} {totals['val']:6}")
        if PROCESSED.exists():
            shutil.rmtree(PROCESSED)
        shutil.copytree(stage, PROCESSED)
    print(f"\nCompleted with seed {SEED}. Original images under {RAW} were left unchanged.")
    print("Note: filenames do not expose leaf/source groups; use dataset_verify.py to audit exact and near duplicates.")


if __name__ == "__main__":
    main()
