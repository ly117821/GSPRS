import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from build_manifests import build_bundle


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTE_KEYS = ("weather", "timeofday", "scene")
COUNTS = {"train": 4537, "val": 1296, "test": 648}


def validate_protocol_config(config):
    if not isinstance(config, dict) or config.get("protocol_identity") != "gsprs-bdd100k-proposed-v1":
        raise ValueError("Unsupported protocol identity.")
    counts = config.get("target_counts")
    required_counts = dict(COUNTS, total=sum(COUNTS.values()))
    if not isinstance(counts, dict) or set(counts) != set(required_counts) or any(type(counts[key]) is not int or counts[key] != value for key, value in required_counts.items()):
        raise ValueError("This protocol requires exactly 4537/1296/648 images and total 6481.")
    dataset = config.get("source_dataset")
    if not isinstance(dataset, dict) or dataset.get("official_source_splits") != ["train"]:
        raise ValueError("This protocol uses source training records only.")
    strata = config.get("stratification")
    required_strata = {
        "attributes": list(ATTRIBUTE_KEYS),
        "joint_or_marginal": "joint",
        "allocation_method": "Hamilton largest remainder with exact integer arithmetic",
        "tie_breaking": "ascending lexicographic stratum tuple",
        "partition_quota_order": ["train_from_selected", "val_from_remaining", "test_is_remainder"],
        "missing_attribute_policy": "Represent missing/null/empty attributes as the empty string; keep them as separate strata.",
        "rare_stratum_policy": "No minimum quota; very small strata may receive zero samples.",
        "allow_missing_attribute_strata": True,
    }
    if not isinstance(strata, dict):
        raise ValueError("Stratification configuration is missing.")
    for key, value in required_strata.items():
        if type(strata.get(key)) is not type(value) or strata.get(key) != value:
            raise ValueError("Configured stratification." + key + " differs from the implemented protocol.")
    for phase in ("sampling", "partitioning"):
        record = config.get(phase)
        if not isinstance(record, dict) or type(record.get("seed")) is not int or record.get("seed_status") != "recorded":
            raise ValueError(phase + " requires a recorded integer seed.")
        if record.get("random_generator") != "SHA-256 seeded deterministic ranking":
            raise ValueError(phase + ".random_generator differs from the implemented protocol.")
        if record.get("library") != "Python standard library hashlib and json":
            raise ValueError(phase + ".library differs from the implemented protocol.")
        expected_order = "Input annotation order is ignored; strata and filenames use lexicographic order." if phase == "sampling" else "Rank selected filenames within each stratum; assign train, then val, then test quotas."
        if record.get("input_order") != expected_order:
            raise ValueError(phase + ".input_order differs from the implemented protocol.")
        if not isinstance(record.get("library_version"), str) or not record["library_version"].strip():
            raise ValueError(phase + ".library_version is missing.")
    payload = 'UTF-8 JSON array [protocol_identity, stage, seed, weather, timeofday, scene, image_filename], ensure_ascii=false, separators=(comma,colon), no whitespace; stage is sample or partition. Sort by digest hex then filename.'
    if config.get("hash_payload") != payload:
        raise ValueError("Configured hash_payload differs from the implemented protocol.")
    if config.get("split_file_order") != "ascending image filename; UTF-8, LF, one original filename per line":
        raise ValueError("Configured split_file_order differs from the implemented protocol.")


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def apportion(sizes, total):

    if any(type(size) is not int or size < 0 for size in sizes.values()):
        raise ValueError("Stratum sizes must be nonnegative integers.")
    population = sum(sizes.values())
    if type(total) is not int or total < 0 or total > population:
        raise ValueError("Requested quota must be between zero and pool size.")
    if population == 0:
        return {key: 0 for key in sizes}
    allocations = {key: size * total // population for key, size in sizes.items()}
    remaining = total - sum(allocations.values())
    order = sorted(sizes, key=lambda key: (-(sizes[key] * total % population), key))
    for key in order[:remaining]:
        allocations[key] += 1
    return allocations


def rank(protocol, stage, seed, stratum, filename):
    payload = [protocol, stage, seed] + list(stratum) + [filename]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), filename


def select_splits(records, *, protocol, sample_seed, partition_seed, counts=None):

    counts = dict(COUNTS if counts is None else counts)
    if set(counts) != {"train", "val", "test"}:
        raise ValueError("Counts must specify train, val and test.")
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("Split counts must be nonnegative integers.")
    if type(sample_seed) is not int or type(partition_seed) is not int:
        raise ValueError("Both seeds must be recorded integers.")
    groups = defaultdict(list)
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every annotation record must be an object.")
        name = record.get("name")
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("Each record needs a nonempty original image filename.")
        if any(char in name for char in ("/", "\\", "\n", "\r", ":")) or name in (".", ".."):
            raise ValueError("Image names must be bare filenames, not filesystem paths.")
        if Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError("Unsupported image filename: " + name)
        if name in seen:
            raise ValueError("Duplicate source filename: " + name)
        seen.add(name)
        attrs = record.get("attributes") or {}
        if not isinstance(attrs, dict):
            raise ValueError("Image attributes must be an object.")
        values = []
        for key in ATTRIBUTE_KEYS:
            value = attrs.get(key)
            value = "" if value is None else value
            if not isinstance(value, str):
                raise ValueError("Image scene attributes must be strings or null.")
            values.append(value)
        groups[tuple(values)].append(name)
    selected_sizes = apportion({key: len(value) for key, value in groups.items()}, sum(counts.values()))
    train_sizes = apportion(selected_sizes, counts["train"])
    remainder_sizes = {key: selected_sizes[key] - train_sizes[key] for key in groups}
    val_sizes = apportion(remainder_sizes, counts["val"])
    output = {key: [] for key in ("train", "val", "test")}
    summary = []
    for key in sorted(groups):
        selected = sorted(groups[key], key=lambda name: rank(protocol, "sample", sample_seed, key, name))[:selected_sizes[key]]
        selected.sort(key=lambda name: rank(protocol, "partition", partition_seed, key, name))
        first = train_sizes[key]
        second = first + val_sizes[key]
        output["train"].extend(selected[:first])
        output["val"].extend(selected[first:second])
        output["test"].extend(selected[second:])
        summary.append(dict(zip(ATTRIBUTE_KEYS, key), source_count=len(groups[key]), selected_count=len(selected), train=first, val=val_sizes[key], test=len(selected)-second))
    for split in output:
        output[split].sort()
    return output, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate BDD100K lists using configured stratification and deterministic ranking. See README.md for the protocol.")
    parser.add_argument("--annotations", type=Path, required=True, help="Official train annotation JSON or documented name/attribute projection of it.")
    parser.add_argument("--annotation-release", required=True, help="Source annotation release description.")
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--source-provenance", type=Path, help="Optional public provenance JSON for a mirrored/derived metadata source.")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "sampling_config.json")
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT / "generated")
    parser.add_argument("--force", action="store_true", help="Replace files in the selected output directory.")
    args = parser.parse_args(argv)
    try:
        if args.output_root.resolve() == PACKAGE_ROOT.resolve():
            raise ValueError("Use a separate output directory for generated materials.")
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        validate_protocol_config(config)
        annotation_digest = sha256_file(args.annotations)
        expected_digest = config.get("source_dataset", {}).get("annotation_sha256")
        if expected_digest and expected_digest != annotation_digest:
            raise ValueError("Annotation SHA-256 differs from the recorded source. Use the provided legacy metadata extractor for this release.")
        records = json.loads(args.annotations.read_text(encoding="utf-8-sig"))
        if isinstance(records, dict):
            records = records.get("frames")
        if not isinstance(records, list):
            raise ValueError("Expected an image-record list or an object with a frames list.")
        expected_pool_count = config.get("source_dataset", {}).get("expected_source_count")
        if expected_pool_count is not None and len(records) != expected_pool_count:
            raise ValueError("Source candidate count differs from the recorded source.")
        outputs, strata = select_splits(records, protocol=config["protocol_identity"], sample_seed=config["sampling"]["seed"], partition_seed=config["partitioning"]["seed"])
        provenance = None
        if args.source_provenance:
            provenance = json.loads(args.source_provenance.read_text(encoding="utf-8-sig"))
            if not isinstance(provenance, dict):
                raise ValueError("Source provenance must be a JSON object.")
        config["source_dataset"].update(annotation_release=args.annotation_release, annotation_filename=args.annotations.name, annotation_sha256=annotation_digest, candidate_count=len(records), source_url=args.source_url)
        if provenance is not None:
            config["source_dataset"]["source_provenance"] = provenance
        build_bundle(split_inputs=outputs, output_root=args.output_root, annotation_files={"train": args.annotations}, source_official_split="train", force=args.force)
        (args.output_root / "configs").mkdir(parents=True, exist_ok=True)
        (args.output_root / "configs" / "sampling_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (args.output_root / "stratum_counts.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ATTRIBUTE_KEYS) + ["source_count", "selected_count", "train", "val", "test"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(strata)
        print("Generated split: 4537 train / 1296 val / 648 test.")
        print("Output:", args.output_root)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
