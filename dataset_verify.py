"""Audit class folders, counts, image readability, label order, and split overlap."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import json
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CLASSES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
SPLITS = ("train", "test", "val")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def image_files(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)


def difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        image = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        values = list(image.getdata())
    result = 0
    for row in range(8):
        for col in range(8):
            result = (result << 1) | int(values[row * 9 + col] > values[row * 9 + col + 1])
    return result


def audit_tree(base: Path, tree_name: str, errors: list[str]):
    counts = Counter()
    hashes = defaultdict(list)
    dhashes = defaultdict(list)
    bad_images = []
    if not base.is_dir():
        errors.append(f"Missing folder: {base}")
        return counts, hashes, dhashes
    actual = sorted(p.name for p in base.iterdir() if p.is_dir())
    if actual != sorted(CLASSES):
        errors.append(f"{tree_name} folders must be exactly {CLASSES}; found {actual}")
    print(f"\n{tree_name} class counts (index follows the ordered labels below):")
    for index, name in enumerate(CLASSES):
        folder = base / name
        if not folder.is_dir():
            errors.append(f"Missing class folder: {folder}")
            continue
        paths = image_files(folder)
        counts[name] = len(paths)
        if not paths:
            errors.append(f"Empty class folder: {folder}")
        print(f"  {index}: {name}: {len(paths)} images")
        for path in paths:
            try:
                with path.open("rb") as stream:
                    digest = sha256(stream.read()).hexdigest()
                d_hash = difference_hash(path)
                hashes[digest].append((tree_name, name, path))
                dhashes[d_hash].append((tree_name, name, path))
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                bad_images.append(f"{path}: {exc}")
    if bad_images:
        errors.extend(f"Unreadable image: {item}" for item in bad_images)
    return counts, hashes, dhashes


def main() -> None:
    errors = []
    raw_counts, raw_hashes, _ = audit_tree(DATA / "raw", "raw", errors)
    split_counts = {}
    all_hashes = defaultdict(list)
    all_dhashes = defaultdict(list)
    for split in SPLITS:
        split_counts[split], hashes, dhashes = audit_tree(DATA / "processed" / split, split, errors)
        for digest, records in hashes.items():
            all_hashes[digest].extend(records)
        for digest, records in dhashes.items():
            all_dhashes[digest].extend(records)

    print("\nSplit totals:")
    for split in SPLITS:
        print(f"  {split}: {sum(split_counts.get(split, {}).values())} images")
    for name in CLASSES:
        actual = [split_counts.get(s, {}).get(name, 0) for s in SPLITS]
        n = raw_counts.get(name, 0)
        exact = [0.6 * n, 0.2 * n, 0.2 * n]
        expected = [int(value) for value in exact]
        order = sorted(range(3), key=lambda i: (exact[i] - expected[i], -i), reverse=True)
        for index in order[:n - sum(expected)]:
            expected[index] += 1
        if actual != expected:
            errors.append(f"{name}: split counts {actual} do not match expected largest-remainder counts {expected}")

    # Distinct source filenames are used by this dataset; byte hashes catch renamed copies.
    cross_split_duplicates = []
    for records in all_hashes.values():
        split_names = {record[0] for record in records}
        if len(split_names) > 1:
            cross_split_duplicates.append(records)
    if cross_split_duplicates:
        errors.append(f"Found {len(cross_split_duplicates)} byte-identical image group(s) across splits")
    else:
        print("Exact duplicate check: no byte-identical images occur across train/test/val.")

    # dHash exact matches are a conservative near-duplicate screen. These are flagged for review,
    # not treated as proof that two photos show the same leaf.
    near = [records for records in all_dhashes.values() if len({r[0] for r in records}) > 1]
    print(f"Exact perceptual-hash groups spanning splits (review candidates): {len(near)}")

    labels_path = ROOT / "models" / "class_names.json"
    if labels_path.is_file():
        try:
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            print(f"Saved model class order: {labels}")
            if labels != CLASSES:
                errors.append(f"{labels_path} class order {labels} does not match {CLASSES}")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Cannot read {labels_path}: {exc}")
    else:
        print("No models/class_names.json yet; train.py will create it in the fixed class order.")

    if errors:
        print("\nAUDIT FAILED:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("\nAUDIT PASSED: class folders, non-empty splits, counts, readable images, and label order are consistent.")


if __name__ == "__main__":
    main()
