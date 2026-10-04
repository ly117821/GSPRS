from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from build_manifests import DATA_PATHS, EXPECTED_COUNTS, FIELDS, HASH_PATH, MANIFEST_PATH, SPLITS, MaterialError, image_filename


def _present(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None and value != "" and value != [] and value != {}


def _load_config(root: Path, errors: list[str], missing: list[str]) -> dict[str, Any] | None:
    candidates = (root / "configs/sampling_config.json", root / "config/sampling_config.json", root / "sampling_config.json")
    target = next((p for p in candidates if p.is_file()), None)
    if target is None:
        missing.append("configs/sampling_config.json is missing.")
        return None
    try:
        config = json.loads(target.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict):
            raise ValueError("root must be an object")
        return config
    except (OSError, ValueError) as exc:
        errors.append(f"Invalid sampling configuration: {exc}")
        return None


def _check_reconstruction(config: dict[str, Any] | None, errors: list[str], missing: list[str], warnings: list[str]) -> dict[str, Any]:
    if config is None:
        return {"status": "INCOMPLETE", "reason": "Sampling configuration unavailable."}
    result: dict[str, Any] = {"status": "DECLARATIONS_COMPLETE", "phases": {}}
    unknown = False
    unavailable = False
    allowed = {"recorded", "not_recorded", "not_applicable", "unknown"}
    for phase in ("sampling", "partitioning"):
        record = config.get(phase)
        if not isinstance(record, dict):
            missing.append(f"{phase} configuration is missing.")
            result["phases"][phase] = "INCOMPLETE"
            unknown = True
            continue
        status = record.get("seed_status", "unknown")
        seed = record.get("seed")
        if not isinstance(status, str) or status not in allowed:
            errors.append(f"{phase}.seed_status must be one of: {', '.join(sorted(allowed))}.")
            unknown = True
        elif status == "recorded":
            if not isinstance(seed, int) or isinstance(seed, bool):
                errors.append(f"{phase}.seed_status is recorded but seed is not an integer.")
                unknown = True
            for key in ("random_generator", "library", "library_version", "input_order"):
                if not _present(record.get(key)):
                    missing.append(f"{phase}.{key} has not been specified.")
                    unknown = True
        elif status == "unknown":
            if seed is not None:
                errors.append(f"{phase}.seed is set while seed_status is unknown; reconcile the declaration.")
            missing.append(f"{phase} seed has not been recorded or explicitly disclosed as unavailable.")
            unknown = True
        elif status == "not_recorded":
            if seed is not None:
                errors.append(f"{phase}.seed must be null when seed_status is not_recorded.")
            unavailable = True
            strategy = record.get("reconstruction_strategy", config.get("reconstruction_strategy"))
            if strategy != "fixed_original_lists":
                missing.append(f"{phase}: a lost seed needs an explicit fixed_original_lists reconstruction strategy.")
                unknown = True
        elif status == "not_applicable":
            if seed is not None:
                errors.append(f"{phase}.seed must be null when seed_status is not_applicable.")
            if not _present(record.get("reason")):
                missing.append(f"{phase}: explain why a random seed is not applicable in the reason field.")
                unknown = True
        result["phases"][phase] = {"seed_status": status, "seed": seed}
    stratification = config.get("stratification")
    if not isinstance(stratification, dict):
        missing.append("stratification configuration has not been specified.")
        unknown = True
    else:
        attributes = stratification.get("attributes")
        if not _present(attributes):
            missing.append("stratification.attributes has not been specified.")
            unknown = True
        elif not isinstance(attributes, list) or len(attributes) != 3 or any(not isinstance(item, str) for item in attributes) or set(attributes) != {"weather", "timeofday", "scene"}:
            errors.append("stratification.attributes must contain exactly weather, timeofday and scene.")
            unknown = True
        mode = stratification.get("joint_or_marginal")
        if not _present(mode):
            missing.append("stratification.joint_or_marginal has not been specified.")
            unknown = True
        elif mode not in ("joint", "marginal"):
            errors.append("stratification.joint_or_marginal must be joint or marginal.")
            unknown = True
        for key in ("allocation_method", "missing_attribute_policy"):
            value = stratification.get(key)
            if not _present(value):
                missing.append(f"stratification.{key} has not been specified.")
                unknown = True
            elif not isinstance(value, str):
                errors.append(f"stratification.{key} must be a nonblank string.")
                unknown = True
    if unavailable:
        warnings.append("At least one historical seed was not recorded. Fixed lists can reproduce membership; the original random construction cannot be replayed.")
        result["status"] = "FIXED_LISTS_ONLY" if not unknown else "INCOMPLETE"
    elif unknown:
        result["status"] = "INCOMPLETE"
    return result


def _check_target_counts(config: dict[str, Any], expected: dict[str, int],
                         split_names: dict[str, list[str]], errors: list[str], missing: list[str]) -> None:
    counts = config.get("target_counts")
    if counts is None:
        missing.append("target_counts has not been specified.")
        return
    if not isinstance(counts, dict):
        errors.append("target_counts must be an object containing total, train, val and test.")
        return
    required = {"total", *SPLITS}
    if set(counts) - required:
        errors.append("target_counts must contain only total, train, val and test.")
    for key in ("total", *SPLITS):
        value = counts.get(key)
        if value is None:
            missing.append(f"target_counts.{key} has not been specified.")
            continue
        if type(value) is not int or value < 0:
            errors.append(f"target_counts.{key} must be a nonnegative integer, not a boolean.")
            continue
        expected_value = sum(expected.values()) if key == "total" else expected[key]
        if value != expected_value:
            errors.append(f"target_counts.{key}: declared {value}, expected {expected_value}.")
        actual = None
        if key == "total" and all(split in split_names for split in SPLITS):
            actual = sum(len(split_names[split]) for split in SPLITS)
        elif key in split_names:
            actual = len(split_names[key])
        if actual is not None and value != actual:
            errors.append(f"target_counts.{key}: declared {value}, actual list count {actual}.")
    if all(type(counts.get(key)) is int for key in required) and counts["total"] != sum(counts[split] for split in SPLITS):
        errors.append("target_counts.total does not equal train + val + test.")


def _check_evaluation_list(path: Path | None, split_names: dict[str, list[str]],
                           errors: list[str], missing: list[str], warnings: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "NOT_CHECKED",
        "reference_sha256": None,
        "reference_count": None,
        "overlap_counts": {split: None for split in SPLITS},
        "overlap_filenames": {split: None for split in SPLITS},
    }
    if path is None:
        return result
    try:
        payload = Path(path).read_bytes()
        lines = payload.decode("utf-8-sig").splitlines()
        if not lines:
            raise MaterialError("Evaluation reference contains no image filenames.")
        names: set[str] = set()
        for line in lines:
            name = image_filename(line)
            if name != line:
                raise MaterialError("Evaluation reference must contain bare filenames, not paths, whitespace or quoted values.")
            folded = name.casefold()
            if folded in names:
                raise MaterialError(f"Duplicate evaluation reference image: {name}")
            names.add(folded)
    except (OSError, ValueError) as exc:
        errors.append(f"Cannot validate evaluation reference: {exc}")
        result["status"] = "INVALID"
        return result
    result["reference_sha256"] = hashlib.sha256(payload).hexdigest()
    result["reference_count"] = len(names)
    result["status"] = "VERIFIED"
    for split in SPLITS:
        if split not in split_names:
            missing.append(f"Evaluation overlap with local {split} cannot be checked because its list is unavailable.")
            if result["status"] != "INVALID":
                result["status"] = "INCOMPLETE"
            continue
        overlaps = sorted(name for name in split_names[split] if name.casefold() in names)
        result["overlap_counts"][split] = len(overlaps)
        result["overlap_filenames"][split] = overlaps
        if overlaps and split in ("train", "val"):
            errors.append(f"Evaluation reference overlaps local {split} by {len(overlaps)} images; evaluation leakage detected.")
            result["status"] = "INVALID"
        elif overlaps:
            warnings.append(f"Evaluation reference overlaps local test by {len(overlaps)} images; the two evaluation sets are not disjoint.")
    return result


def verify_bundle(bundle_root: Path, images_root: Path | None = None,
                  expected_counts: dict[str, int] | None = None, *,
                  evaluation_list: Path | None = None) -> dict[str, Any]:
    root = Path(bundle_root)
    expected = EXPECTED_COUNTS if expected_counts is None else expected_counts
    errors: list[str] = []
    missing: list[str] = []
    warnings: list[str] = []
    partition_errors: list[str] = []
    partition_missing: list[str] = []
    split_names: dict[str, list[str]] = {}
    membership: dict[str, str] = {}
    case_membership: dict[str, str] = {}
    for split in SPLITS:
        relative = f"splits/bdd100k/{split}.txt"
        path = root / relative
        if not path.is_file():
            partition_missing.append(f"{relative} is missing; real image membership is required.")
            continue
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
            names = []
            for line in lines:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                name = image_filename(line)
                if line != name:
                    raise MaterialError(f"Published lists must contain bare filenames, not paths or quoted values ({name}).")
                names.append(name)
            split_names[split] = names
            if not names:
                partition_missing.append(f"{relative} contains no image filenames.")
            elif len(names) != expected[split]:
                partition_errors.append(f"{relative}: expected {expected[split]} images, found {len(names)}.")
            for name in names:
                folded = name.casefold()
                if folded in case_membership:
                    partition_errors.append(f"Duplicate or overlapping image: {name} in {case_membership[folded]} and {split}.")
                case_membership[folded] = split
                membership[name] = split
        except (OSError, ValueError) as exc:
            partition_errors.append(f"Cannot validate {relative}: {exc}")

    rows: list[dict[str, str]] = []
    manifest_path = root / MANIFEST_PATH
    if not manifest_path.is_file():
        partition_missing.append(f"{MANIFEST_PATH} is missing.")
    else:
        try:
            reader = csv.DictReader(io.StringIO(manifest_path.read_text(encoding="utf-8-sig")))
            if reader.fieldnames != list(FIELDS):
                raise MaterialError("Manifest columns must be: " + ",".join(FIELDS))
            rows = list(reader)
            if not rows:
                partition_missing.append("The manifest has no image records.")
        except (OSError, ValueError, csv.Error) as exc:
            partition_errors.append(f"Cannot validate manifest: {exc}")

    manifest_names: set[str] = set()
    sources: set[str] = set()
    missing_source_count = 0
    missing_attribute_count = 0
    missing_images: list[str] = []
    image_checks = 0
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            partition_errors.append("Manifest contains a row with the wrong number of fields.")
            continue
        name = row.get("image_filename", "")
        try:
            if image_filename(name) != name:
                raise MaterialError("Manifest image_filename must be a bare filename.")
        except MaterialError as exc:
            partition_errors.append(str(exc))
            continue
        if name in manifest_names:
            partition_errors.append(f"Duplicate manifest record: {name}")
        manifest_names.add(name)
        if row["image_id"] != Path(name).stem:
            partition_errors.append(f"Manifest image_id does not match image_filename: {name}")
        split = row["local_split"]
        if split not in SPLITS or membership.get(name) != split:
            partition_errors.append(f"Manifest/list membership mismatch: {name} ({split}).")
        source = row["source_official_split"]
        relative = row["relative_path"]
        if source not in {"", "train", "val"}:
            partition_errors.append(f"Unsupported official source for {name}: {source}")
        elif source:
            sources.add(source)
            if relative != f"images/100k/{source}/{name}":
                partition_errors.append(f"Official relative path disagrees with declared source: {name}")
        else:
            missing_source_count += 1
            if relative:
                partition_errors.append(f"Official relative path is present without a declared official source: {name}")
        if not all(row[key] for key in ("weather", "timeofday", "scene")):
            missing_attribute_count += 1
        if images_root is not None and source in {"train", "val"} and relative == f"images/100k/{source}/{name}":
            image_checks += 1
            if not (Path(images_root) / relative).is_file():
                missing_images.append(relative)
    if rows and membership and manifest_names != set(membership):
        partition_errors.append(f"Manifest/list image sets differ: {len(set(membership) - manifest_names)} missing and {len(manifest_names - set(membership))} extra manifest records.")
    if missing_source_count:
        missing.append(f"Official source split is unspecified for {missing_source_count} images.")
    if missing_images:
        errors.append(f"{len(missing_images)} declared image files were not found under --images-root (expected a BDD100K dataset root).")
    if images_root is not None and missing_source_count:
        warnings.append("Images with unknown official source could not be checked on disk.")

    hash_status = "VERIFIED"
    actual_hashes: dict[str, str] = {}
    if not (root / HASH_PATH).is_file():
        partition_missing.append(f"{HASH_PATH} is missing.")
        hash_status = "INCOMPLETE"
    else:
        try:
            hash_document = json.loads((root / HASH_PATH).read_text(encoding="utf-8-sig"))
            if not isinstance(hash_document, dict) or hash_document.get("algorithm") != "sha256" or not isinstance(hash_document.get("files"), dict):
                raise MaterialError("Expected an object with algorithm=sha256 and a files hash mapping.")
            hashes = hash_document["files"]
            if set(hashes) != set(DATA_PATHS):
                raise MaterialError("SHA256SUMS.json must contain exactly the three split lists and the manifest.")
            for relative in DATA_PATHS:
                if not (root / relative).is_file():
                    hash_status = "INCOMPLETE"
                    continue
                digest = hashes[relative]
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise MaterialError(f"Invalid SHA-256 value for {relative}.")
                actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
                actual_hashes[relative] = actual
                if actual != digest:
                    partition_errors.append(f"SHA-256 mismatch: {relative}")
                    hash_status = "INVALID"
        except (OSError, ValueError) as exc:
            partition_errors.append(f"Cannot validate {HASH_PATH}: {exc}")
            hash_status = "INVALID"

    config = _load_config(root, errors, missing)
    if missing_attribute_count:
        stratification = config.get("stratification", {}) if config is not None else {}
        allow_missing_attributes = (
            isinstance(stratification, dict)
            and stratification.get("allow_missing_attribute_strata") is True
            and _present(stratification.get("missing_attribute_policy"))
        )
        if allow_missing_attributes:
            warnings.append(f"Weather/timeofday/scene metadata is missing for {missing_attribute_count} images; empty values are preserved as strata under the explicitly declared missing-attribute policy.")
        else:
            missing.append(f"Weather/timeofday/scene metadata is incomplete for {missing_attribute_count} images.")
    reconstruction = _check_reconstruction(config, errors, missing, warnings)
    if config is not None:
        _check_target_counts(config, expected, split_names, errors, missing)
        dataset = config.get("source_dataset")
        if not isinstance(dataset, dict):
            missing.append("source_dataset configuration is missing.")
        else:
            declared_sources = dataset.get("official_source_splits")
            if not isinstance(declared_sources, list) or not declared_sources:
                missing.append("source_dataset.official_source_splits has not been specified.")
            elif any(not isinstance(item, str) or item not in {"train", "val"} for item in declared_sources):
                errors.append("source_dataset.official_source_splits must contain only train and/or val.")
            elif rows and sources and set(declared_sources) != sources:
                errors.append("Declared official source splits do not match the manifest.")
            if not _present(dataset.get("annotation_release")):
                missing.append("source_dataset.annotation_release has not been specified.")
    evaluation_audit = _check_evaluation_list(evaluation_list, split_names, errors, missing, warnings)
    errors = partition_errors + errors
    missing = partition_missing + missing
    partition_status = "INVALID" if partition_errors else "INCOMPLETE" if partition_missing else "VERIFIED"
    local_partition_sha256 = None
    if partition_status == "VERIFIED" and hash_status == "VERIFIED" and set(actual_hashes) == set(DATA_PATHS):
        canonical_hashes = json.dumps(actual_hashes, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        local_partition_sha256 = hashlib.sha256(canonical_hashes).hexdigest()
    status = "INVALID" if errors else "INCOMPLETE" if missing else "PASS_WITH_LIMITATIONS" if reconstruction["status"] == "FIXED_LISTS_ONLY" else "PASS"
    return {
        "schema_version": 1,
        "status": status,
        "exit_code": 1 if errors else 2 if missing else 0,
        "protocol_identity": config.get("protocol_identity") if config is not None else None,
        "local_partition_sha256": local_partition_sha256,
        "partition": {
            "status": partition_status,
            "expected_counts": expected,
            "actual_counts": {split: len(split_names[split]) if split in split_names else None for split in SPLITS},
            "manifest_rows": len(rows),
            "hash_status": hash_status,
            "official_source_splits": sorted(sources),
            "missing_source_count": missing_source_count,
            "missing_attribute_count": missing_attribute_count,
        },
        "sampling_process": reconstruction,
        "evaluation_audit": evaluation_audit,
        "images": {"checked": images_root is not None, "files_checked": image_checks, "missing_count": len(missing_images), "missing_examples": missing_images[:10]},
        "errors": errors,
        "missing_materials": missing,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check partition membership, metadata, hashes, and configuration. Exit codes: 0 = valid; 1 = invalid; 2 = incomplete.")
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--images-root", type=Path, help="Dataset root containing images/100k/train and/or images/100k/val.")
    parser.add_argument("--evaluation-list", type=Path, help="Actual supplementary evaluation reference: one bare image filename per line; audit local train, val and test overlap.")
    parser.add_argument("--write-report", type=Path, help="Write the same JSON report to this file.")
    args = parser.parse_args()
    report = verify_bundle(args.bundle_root, args.images_root, evaluation_list=args.evaluation_list)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        with args.write_report.open("w", encoding="utf-8", newline="\n") as report_file:
            report_file.write(rendered)
    print(rendered, end="")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
