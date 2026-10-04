# BDD100K Partition Materials

This directory provides the exact BDD100K local partition used in the GSPRS experiments. It contains the image lists, partition manifest, deterministic sampling configuration, source metadata, and verification tools. It does not redistribute the original BDD100K images or annotations.

## Partition

| Split | Images | File |
| --- | ---: | --- |
| Training | 4,537 | `splits/bdd100k/train.txt` |
| Validation | 1,296 | `splits/bdd100k/val.txt` |
| Local test | 648 | `splits/bdd100k/test.txt` |
| Total | 6,481 | `data/bdd100k/subset_manifest.csv` |

The lists contain one image filename per line. The manifest records the source partition, local assignment, relative image path, weather, time of day, and scene type. The local partition is sampled from the BDD100K training source and does not overlap the 10,000-image source validation list stored under `evaluation/`.

## Deterministic construction

The released configuration uses a joint weather, time-of-day, and scene stratification scheme. Integer quotas are allocated with Hamilton's largest-remainder rule. The sampling seed is **314159**, and the partition-assignment seed is **271828**. The first seed selects the 6,481-image subset from the 70,000 source training records; the second assigns the selected images to the training, validation, and local-test splits. These are data-selection seeds, not model-training seeds.

The implementation uses deterministic SHA-256 ranking with the canonical payload recorded in `configs/sampling_config.json`. Reusing the pinned source metadata, configuration, and seeds regenerates the released image lists. The image lists are the authoritative record of the partition used in the experiments.

The configuration identifier `gsprs-bdd100k-proposed-v1` is retained for byte-level reproducibility because it is part of the ranking payload. It should not be changed when replaying the partition.

The source is the [BDD100K legacy label archive](https://archive.org/download/bdd100k/bdd100k_labels.zip). Source provenance and metadata checksums are recorded under `docs/`. The original images and full annotations must be obtained from the dataset provider and are not redistributed here.

## Verification

The utilities require Python 3.9 or later and use the standard library. From this directory, run:

```console
python scripts/verify_bundle.py --evaluation-list evaluation/bdd100k_full_validation.txt --write-report verification_report.json
python -m unittest discover -s tests -v
```

The verifier checks split counts, duplicates, overlap, manifest membership, source attributes, file hashes, and overlap with the supplied source-validation list. It verifies partition consistency; it does not reproduce model accuracy.

`SHA256SUMS.json` protects the released lists and manifest. `verification_report.json` records the verified partition fingerprint, and `replay_report.json` records the deterministic reconstruction check.

## Repository

The project repository is available at:

https://github.com/ly117821/GSPRS

The detector implementation and training code will be released separately after publication. This directory is the reproducibility release for the BDD100K partition protocol.
