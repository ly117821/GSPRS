import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile


ARCHIVE_SHA256 = "a8bed9c8f4371f85a57cbd7aec5d7d40f5723cecfa611de0923e01a18c39822a"
ARCHIVE_SIZE = 154084208
SOURCE_URL = "https://archive.org/download/bdd100k/bdd100k_labels.zip"
ATTRIBUTES = ("weather", "timeofday", "scene")
EXPECTED = {
    "train": (70000, "3520766b3ef93ed46986c280dc913a0d9a8bc3cb14f4103461ba76d64fc8865c"),
    "val": (10000, "f4c7a404197d723e804c78b5b366b1bcc4fd166a1a9c6a3996a19b3251eb7148"),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_split(archive, split):

    prefix = "bdd100k/labels/100k/{}/".format(split)
    members = sorted(
        name for name in archive.namelist()
        if name.startswith(prefix) and name.endswith(".json")
    )
    records = []
    seen = set()
    for member in members:
        source = json.loads(archive.read(member))
        name = source["name"]


        if not name.endswith(".jpg"):
            name += ".jpg"
        if name in seen:
            raise ValueError("Duplicate image name in {}: {}".format(split, name))
        seen.add(name)
        attrs = source.get("attributes", {})
        if any(not isinstance(attrs.get(key), str) for key in ATTRIBUTES):
            raise ValueError("Missing/non-string frame attributes: " + member)
        records.append({
            "name": name,
            "attributes": {key: attrs.get(key) for key in ATTRIBUTES},
            "source_split": split,
            "source_member": member,
        })
    if len(records) != EXPECTED[split][0]:
        raise ValueError("Unexpected {} record count: {}".format(split, len(records)))


    payload = (json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\r\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED[split][1]:
        raise ValueError("Projection checksum mismatch for " + split)
    return payload, seen, digest


def run(archive_path, output_dir):
    if archive_path.stat().st_size != ARCHIVE_SIZE or file_sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError("Input does not match the specific 2018 mirror archive used by this package.")
    outputs = [output_dir / (split + "_metadata.json") for split in EXPECTED]
    outputs.append(output_dir / "provenance.json")
    if any(path.exists() for path in outputs):
        raise ValueError("Output files already exist; select a fresh --output-dir.")
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "schema_version": "1.0",
        "source_url": SOURCE_URL,
        "source_kind": "public Internet Archive mirror of legacy 2018 per-image labels",
        "source_metadata_url": "https://archive.org/metadata/bdd100k",
        "archive_sha256": ARCHIVE_SHA256,
        "archive_size_bytes": ARCHIVE_SIZE,
        "name_transformation": "Append .jpg when absent in source JSON name.",
        "projection": "Only name, weather/timeofday/scene attributes, source_split, and source_member are retained; frames/object annotations are omitted.",
        "serialization": "Member paths sorted lexicographically; JSON object keys sorted; compact comma/colon separators; UTF-8; one terminal CRLF written explicitly on every platform.",
        "outputs": {},
    }
    materialized = {}
    names = {}
    with zipfile.ZipFile(archive_path) as archive:
        for split in EXPECTED:
            payload, names[split], digest = project_split(archive, split)
            filename = split + "_metadata.json"
            materialized[filename] = payload
            provenance["outputs"][filename] = {
                "records": len(names[split]), "unique_names": len(names[split]),
                "sha256": digest, "size_bytes": len(payload),
            }
    overlap = names["train"] & names["val"]
    if overlap:
        raise ValueError("Train/validation source names overlap.")
    provenance["train_val_overlap_count"] = len(overlap)
    materialized["provenance.json"] = (json.dumps(
        provenance, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n").encode("utf-8")
    for filename, payload in materialized.items():
        (output_dir / filename).write_bytes(payload)
    return provenance


def main():
    parser = argparse.ArgumentParser(description="Extract deterministic image names and scene attributes from the pinned legacy label archive.")
    parser.add_argument("--archive", required=True, type=Path, help="Downloaded bdd100k_labels.zip (2018 mirror)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory with no existing output JSON files")
    args = parser.parse_args()
    try:
        provenance = run(args.archive, args.output_dir)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, "ERROR: {}\n".format(error))
    print(json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
