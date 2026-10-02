# BDD100K Subset Partition Materials

This directory provides the fixed BDD100K local partition for the GSPRS study. It contains image lists, image-level scene attributes, the sampling configuration, source provenance, and partition-checking tools. It does not contain detector weights, repeated-training metrics, or the challenging-subset results in the manuscript.

## Partition

| Split | Images | File |
| --- | ---: | --- |
| Training | 4,537 | `splits/bdd100k/train.txt` |
| Validation | 1,296 | `splits/bdd100k/val.txt` |
| Local test | 648 | `splits/bdd100k/test.txt` |
| Total | 6,481 | `data/bdd100k/subset_manifest.csv` |

Each image list contains one filename per line. The manifest records the source partition, local assignment, relative image path, weather, time of day, and scene type. The local lists contain no duplicate or overlapping filenames. `evaluation/bdd100k_full_validation.txt` records the 10,000 source-validation filenames; none overlaps the local partition.

## Configuration and Source

`configs/sampling_config.json` is the single sampling configuration. Sampling and partitioning use seeds **20261001** and **20261002**, respectively. These are data-selection seeds, not model-training seeds.

The sampler uses only source-training records. Joint weather, time-of-day, and scene strata receive proportional integer quotas using Hamilton largest remainders and lexicographic tie-breaking. Within each stratum, seeded SHA-256 ranking selects images and assigns the local splits. The algorithm identifier `gsprs-bdd100k-proposed-v1` is retained because it is part of the hash payload; changing it changes the selected filenames.

The source is the [Internet Archive mirror of the legacy 2018 BDD100K label archive](https://archive.org/download/bdd100k/bdd100k_labels.zip). Its byte size, archive checksum, extraction procedure, and metadata checksums are recorded in `docs/source_provenance.json`. The source contains 70,000 training and 10,000 validation annotation records. `docs/source_metadata_summary.json` reports attributes of this source, not the manuscript's condition-specific evaluation groups.

Original images and full object annotations are not redistributed here. Obtain dataset files from the dataset provider and follow the applicable dataset terms. The source notice is retained in `LICENSES/BDD100K_2018.txt`.

## Verify the Published Partition

The utilities require Python 3.9 or later and use the standard library. Run from this directory:

```console
python scripts/verify_bundle.py --evaluation-list evaluation/bdd100k_full_validation.txt
python -m unittest discover -s tests -v
```

The verifier checks split sizes, duplicate and overlapping filenames, manifest membership, metadata declarations, file checksums, and overlap with the supplied evaluation list. Exit codes are 0 for valid files, 1 for inconsistent files, and 2 for incomplete inputs. It checks file consistency, not model accuracy. To additionally check image-file existence, supply `--images-root` pointing to the dataset root containing `images/100k/train` and `images/100k/val`.

`SHA256SUMS.json` protects the three local lists and manifest. `PACKAGE_SHA256SUMS.json` records the remaining published files as well. `verification_report.json` is the partition-checking report for this release.

Git attributes preserve the release files byte for byte so that checkout line-ending conversion does not invalidate the checksums.

## Replay the Sampling Procedure

Download the pinned label archive linked above. Extract the deterministic name-and-attribute metadata and generate the partition into a separate directory:

```console
python scripts/extract_legacy_metadata.py --archive /path/to/bdd100k_labels.zip --output-dir metadata
python scripts/sample_proposed_split.py --annotations metadata/train_metadata.json --annotation-release "BDD100K legacy 2018 per-image labels, Internet Archive mirror" --source-url https://archive.org/download/bdd100k/bdd100k_labels.zip --source-provenance metadata/provenance.json --output-root generated
python scripts/verify_bundle.py --bundle-root generated --evaluation-list evaluation/bdd100k_full_validation.txt
```

The extractor rejects archives that differ from the pinned source. The sampler checks the source metadata checksum and generates the three lists and manifest from the recorded configuration. Do not change the algorithm identifier, seeds, annotation release, ranking payload, or quota rules when replaying this partition.
