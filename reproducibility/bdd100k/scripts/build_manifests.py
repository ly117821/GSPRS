from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PureWindowsPath
import tempfile
from typing import Any

EXPECTED_COUNTS = {"train": 4537, "val": 1296, "test": 648}
SPLITS = tuple(EXPECTED_COUNTS)
FIELDS = (
    "image_id", "image_filename", "relative_path", "source_official_split",
    "local_split", "weather", "timeofday", "scene",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MANIFEST_PATH = "data/bdd100k/subset_manifest.csv"
HASH_PATH = "SHA256SUMS.json"
DATA_PATHS = tuple(f"splits/bdd100k/{split}.txt" for split in SPLITS) + (MANIFEST_PATH,)


class MaterialError(ValueError):
    pass


def image_filename(value: str) -> str:

    if not isinstance(value, str):
        raise MaterialError("Image identifiers must be strings.")
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    if not value or any(ord(c) < 32 for c in value):
        raise MaterialError("Empty image identifier or control character in image path.")
    path = PureWindowsPath(value)
    if ".." in path.parts or value.endswith(("/", "\\")):
        raise MaterialError(f"Parent traversal or missing filename: {value!r}")
    name = path.name
    if not name or any(c in name for c in '<>:"/\\|?*') or name.endswith((" ", ".")):
        raise MaterialError(f"Invalid image filename: {name!r}")
    if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
        raise MaterialError(f"Expected an image filename (.jpg, .jpeg, .png): {name!r}")
    if Path(name).stem.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise MaterialError(f"Reserved filename: {name!r}")
    return name


def read_image_list(path: Path) -> list[str]:

    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    return [image_filename(line) for line in lines if line.strip() and not line.lstrip().startswith("#")]


def read_image_directory(path: Path) -> list[str]:

    path = Path(path)
    if not path.is_dir():
        raise MaterialError(f"Image directory does not exist: {path}")
    files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort(key=lambda p: p.relative_to(path).as_posix())
    return [image_filename(p.name) for p in files]


def load_annotations(annotation_files: dict[str, Path] | None = None) -> dict[str, dict[str, Any]]:

    result: dict[str, dict[str, Any]] = {}
    for source, filename in (annotation_files or {}).items():
        if source not in {"train", "val"}:
            raise MaterialError("Official annotation source must be 'train' or 'val'.")
        records = json.loads(Path(filename).read_text(encoding="utf-8-sig"))
        if isinstance(records, dict):
            records = records.get("frames")
        if not isinstance(records, list):
            raise MaterialError(f"{filename}: expected a BDD100K JSON list or an object with a frames array.")
        for record in records:
            if not isinstance(record, dict) or "name" not in record:
                raise MaterialError(f"{filename}: every record must have an image name.")
            name = image_filename(record["name"])
            attrs = record.get("attributes") or {}
            if not isinstance(attrs, dict):
                raise MaterialError(f"{filename}: attributes must be an object for {name}.")
            metadata = {"source_official_split": source}
            for key in ("weather", "timeofday", "scene"):
                value = attrs.get(key, "")
                if value is None:
                    value = ""
                if not isinstance(value, str):
                    raise MaterialError(f"{filename}: {key} must be a string for {name}.")
                metadata[key] = value
            if name in result:
                raise MaterialError(f"Duplicate or ambiguous official annotation record: {name}")
            result[name] = metadata
    return result


def validate_membership(split_inputs: dict[str, list[str]], expected_counts: dict[str, int] | None = None) -> dict[str, list[str]]:
    expected = EXPECTED_COUNTS if expected_counts is None else expected_counts
    if set(split_inputs) != set(SPLITS) or set(expected) != set(SPLITS):
        raise MaterialError("Exactly train, val and test partitions are required.")
    normalized: dict[str, list[str]] = {}
    seen: dict[str, str] = {}
    for split in SPLITS:
        names = [image_filename(item) for item in split_inputs[split]]
        if len(names) != expected[split]:
            raise MaterialError(f"{split}: expected {expected[split]} images; received {len(names)}.")
        for name in names:

            key = name.casefold()
            if key in seen:
                raise MaterialError(f"Duplicate image or split overlap: {name} occurs in {seen[key]} and {split}.")
            seen[key] = split
        normalized[split] = names
    return normalized


def build_bundle(*, split_inputs: dict[str, list[str]], output_root: Path,
                 annotation_files: dict[str, Path] | None = None,
                 source_official_split: str | None = None,
                 expected_counts: dict[str, int] | None = None,
                 force: bool = False) -> dict[str, Any]:

    normalized = validate_membership(split_inputs, expected_counts)
    if source_official_split not in {None, "train", "val"}:
        raise MaterialError("Explicit official source must be 'train' or 'val'.")
    annotation_index = load_annotations(annotation_files)
    rows = []
    missing_sources = 0
    missing_attributes = 0
    for split in SPLITS:
        for name in normalized[split]:
            metadata = annotation_index.get(name, {})
            source = metadata.get("source_official_split", source_official_split or "")
            if metadata and source_official_split and source != source_official_split:
                raise MaterialError(f"Declared source {source_official_split} contradicts annotations for {name} ({source}).")

            if source and source in (annotation_files or {}) and not metadata:
                raise MaterialError(f"{name} was not found in the supplied official {source} annotations.")
            attrs = {key: metadata.get(key, "") for key in ("weather", "timeofday", "scene")}
            if not source:
                missing_sources += 1
            if not all(attrs.values()):
                missing_attributes += 1
            rows.append({
                "image_id": Path(name).stem,
                "image_filename": name,
                "relative_path": f"images/100k/{source}/{name}" if source else "",
                "source_official_split": source,
                "local_split": split,
                **attrs,
            })

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payloads = {f"splits/bdd100k/{split}.txt": ("\n".join(normalized[split]) + "\n").encode("utf-8") for split in SPLITS}
    payloads[MANIFEST_PATH] = buffer.getvalue().encode("utf-8")
    hash_document = {"algorithm": "sha256", "files": {path: hashlib.sha256(payload).hexdigest() for path, payload in payloads.items()}}
    payloads[HASH_PATH] = (json.dumps(hash_document, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    output_root = Path(output_root)
    existing = [path for path in payloads if (output_root / path).exists()]
    if existing and not force:
        raise MaterialError("Generated materials already exist; use --force to replace them: " + ", ".join(existing))
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".manifest-stage-", dir=output_root) as stage:
        staging = Path(stage)
        for relative, payload in payloads.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        for relative in payloads:
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            (staging / relative).replace(destination)
    return {
        "counts": {split: len(normalized[split]) for split in SPLITS},
        "total_images": len(rows),
        "missing_source_count": missing_sources,
        "missing_attribute_count": missing_attributes,
        "files": list(payloads),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Import image lists and annotations into a partition manifest. Requires Python 3.9+ and the standard library.")
    for split in SPLITS:
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(f"--{split}-list", type=Path, help="Text file: one original image filename or path per line.")
        group.add_argument(f"--{split}-dir", type=Path, help="Directory containing this local partition's images.")
    parser.add_argument("--train-annotations", type=Path, help="Original official training annotation JSON list.")
    parser.add_argument("--val-annotations", type=Path, help="Original official validation annotation JSON list.")
    parser.add_argument("--source-official-split", choices=["train", "val"], help="Explicit assertion that ALL imported images originate from this official split.")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true", help="Replace existing generated lists, manifest and hashes.")
    args = parser.parse_args()
    try:
        inputs = {}
        for split in SPLITS:
            list_path = getattr(args, f"{split}_list")
            inputs[split] = read_image_list(list_path) if list_path else read_image_directory(getattr(args, f"{split}_dir"))
        annotations = {split: getattr(args, f"{split}_annotations") for split in ("train", "val") if getattr(args, f"{split}_annotations")}
        result = build_bundle(split_inputs=inputs, output_root=args.output_root, annotation_files=annotations,
                              source_official_split=args.source_official_split, force=args.force)
    except (MaterialError, OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Next: record only supported facts in configs/sampling_config.json, then run scripts/verify_bundle.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
