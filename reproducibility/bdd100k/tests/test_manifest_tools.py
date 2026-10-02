import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_manifests import DATA_PATHS, MaterialError, build_bundle, image_filename, read_image_directory
from verify_bundle import verify_bundle


class ManifestToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.counts = {"train": 2, "val": 1, "test": 1}
        self.inputs = {"train": [r"C:\private\example-a.jpg", "example-b.jpg"], "val": ["example-c.jpg"], "test": ["example-d.jpg"]}
        self.annotations = self.root / "official_train.json"
        self.annotations.write_text(json.dumps([
            {"name": name, "attributes": {"weather": "clear", "timeofday": "daytime", "scene": "city street"}}
            for name in ("example-a.jpg", "example-b.jpg", "example-c.jpg", "example-d.jpg")
        ]), encoding="utf-8")
        self.config = {
            "target_counts": {**self.counts, "total": sum(self.counts.values())},
            "source_dataset": {"official_source_splits": ["train"], "annotation_release": "synthetic-test-only"},
            "sampling": {"seed": 1, "seed_status": "recorded", "random_generator": "Random", "library": "python", "library_version": "test-only", "input_order": "sorted filename"},
            "partitioning": {"seed": 2, "seed_status": "recorded", "random_generator": "Random", "library": "python", "library_version": "test-only", "input_order": "sorted filename"},
            "stratification": {
                "attributes": ["weather", "timeofday", "scene"],
                "joint_or_marginal": "joint",
                "allocation_method": "synthetic-test-only",
                "missing_attribute_policy": "Reject missing attributes in this test fixture.",
            },
        }

    def tearDown(self):
        self.temp.cleanup()

    def write_config(self):
        (self.root / "configs").mkdir(exist_ok=True)
        (self.root / "configs/sampling_config.json").write_text(json.dumps(self.config), encoding="utf-8")

    def build(self, **overrides):
        kwargs = {"split_inputs": self.inputs, "output_root": self.root, "annotation_files": {"train": self.annotations}, "expected_counts": self.counts}
        kwargs.update(overrides)
        return build_bundle(**kwargs)

    def test_successful_round_trip_and_private_paths_removed(self):
        self.build()
        self.write_config()
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["partition"]["status"], "VERIFIED")
        self.assertEqual(report["sampling_process"]["status"], "DECLARATIONS_COMPLETE")
        self.assertEqual(report["evaluation_audit"]["status"], "NOT_CHECKED")
        content = (self.root / "data/bdd100k/subset_manifest.csv").read_text(encoding="utf-8")
        self.assertNotIn("private", content)
        rows = list(csv.DictReader(content.splitlines()))
        self.assertEqual(rows[0]["relative_path"], "images/100k/train/example-a.jpg")
        self.assertEqual(rows[0]["weather"], "clear")

    def test_overlap_is_rejected_without_writing(self):
        self.inputs["test"] = ["example-b.jpg"]
        with self.assertRaisesRegex(MaterialError, "overlap"):
            self.build()
        self.assertFalse((self.root / "splits").exists())

    def test_wrong_count_is_rejected(self):
        self.inputs["train"].pop()
        with self.assertRaisesRegex(MaterialError, "expected 2"):
            self.build()

    def test_hash_tampering_is_detected(self):
        self.build()
        self.write_config()
        path = self.root / "splits/bdd100k/test.txt"
        path.write_bytes(path.read_bytes() + b"\n")
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["status"], "INVALID")
        self.assertIsNone(report["local_partition_sha256"])
        self.assertTrue(any("SHA-256 mismatch" in error for error in report["errors"]))

    def test_unknown_seed_is_incomplete_but_membership_verifies(self):
        self.build()
        self.config["sampling"].update(seed=None, seed_status="unknown")
        self.write_config()
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["exit_code"], 2)
        self.assertEqual(report["partition"]["status"], "VERIFIED")

    def test_disclosed_lost_seed_preserves_fixed_list_verification(self):
        self.build()
        for phase in ("sampling", "partitioning"):
            self.config[phase].update(seed=None, seed_status="not_recorded")
        self.config["reconstruction_strategy"] = "fixed_original_lists"
        self.write_config()
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["status"], "PASS_WITH_LIMITATIONS", report)
        self.assertEqual(report["sampling_process"]["status"], "FIXED_LISTS_ONLY")
        self.assertEqual(report["partition"]["status"], "VERIFIED")

    def test_empty_bundle_is_incomplete_without_crashing(self):
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["exit_code"], 2)

    def test_unknown_source_is_not_inferred_from_local_split(self):
        self.build(annotation_files={})
        self.write_config()
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["partition"]["missing_source_count"], 4)
        self.assertEqual(report["status"], "INCOMPLETE")
        rows = list(csv.DictReader((self.root / "data/bdd100k/subset_manifest.csv").read_text(encoding="utf-8").splitlines()))
        self.assertTrue(all(row["source_official_split"] == row["relative_path"] == "" for row in rows))

    def test_conflicting_source_declaration_is_rejected(self):
        with self.assertRaisesRegex(MaterialError, "contradicts"):
            self.build(source_official_split="val")

    def test_directory_input_and_unsafe_names(self):
        directory = self.root / "images"
        directory.mkdir()
        (directory / "b.jpg").write_bytes(b"")
        (directory / "a.jpg").write_bytes(b"")
        self.assertEqual(read_image_directory(directory), ["a.jpg", "b.jpg"])
        for value in ("../a.jpg", "a?.jpg", "no-image.txt", "C:\\folder\\", "bad\nname.jpg"):
            with self.subTest(value=value), self.assertRaises(MaterialError):
                image_filename(value)

    def test_existing_materials_require_force(self):
        self.build()
        with self.assertRaisesRegex(MaterialError, "--force"):
            self.build()
        self.build(force=True)

    def test_frames_annotation_object_preserves_metadata(self):
        records = json.loads(self.annotations.read_text(encoding="utf-8"))
        self.annotations.write_text(json.dumps({"frames": records}), encoding="utf-8")
        self.build()
        self.write_config()
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["partition"]["missing_attribute_count"], 0)

    def test_invalid_frames_object_is_rejected(self):
        self.annotations.write_text(json.dumps({"frames": {}}), encoding="utf-8")
        with self.assertRaisesRegex(MaterialError, "frames array"):
            self.build()

    def test_missing_attributes_need_an_explicit_stratum_policy(self):
        records = json.loads(self.annotations.read_text(encoding="utf-8"))
        records[0]["attributes"].pop("weather")
        self.annotations.write_text(json.dumps(records), encoding="utf-8")
        self.build()
        self.write_config()
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.config["stratification"].update(
            allow_missing_attribute_strata=True,
            missing_attribute_policy="Preserve empty values as separate strata.",
        )
        self.write_config()
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["partition"]["missing_attribute_count"], 1)
        self.assertTrue(any("empty values are preserved as strata" in warning for warning in report["warnings"]))

    def test_protocol_identity_is_explicit_in_pass_report(self):
        self.build()
        self.config.update(protocol_identity="test-protocol-v1")
        self.write_config()
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["protocol_identity"], self.config["protocol_identity"])
        self.assertEqual(report["partition"]["actual_counts"], self.counts)

    def test_declared_counts_must_match_expected_and_actual_lists(self):
        self.build()
        self.config["target_counts"].update(train=1, val=2)
        self.write_config()
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["status"], "INVALID", report)
        self.assertTrue(any("target_counts.train: declared 1, expected 2" in error for error in report["errors"]))
        self.assertTrue(any("actual list count" in error for error in report["errors"]))

    def test_declared_total_must_equal_partition_sum(self):
        self.build()
        self.config["target_counts"]["total"] = 5
        self.write_config()
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["status"], "INVALID", report)
        self.assertIn("target_counts.total does not equal train + val + test.", report["errors"])

    def test_declared_counts_reject_booleans_floats_and_extra_keys(self):
        self.build()
        for replacement in (True, 1.0, -1, "1"):
            with self.subTest(replacement=replacement):
                self.config["target_counts"]["val"] = replacement
                self.write_config()
                report = verify_bundle(self.root, expected_counts=self.counts)
                self.assertEqual(report["status"], "INVALID", report)
                self.assertTrue(any("target_counts.val must be a nonnegative integer" in error for error in report["errors"]))
        self.config["target_counts"].update(val=1, other=0)
        self.write_config()
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["status"], "INVALID", report)

    def test_missing_declared_count_is_incomplete(self):
        self.build()
        del self.config["target_counts"]["total"]
        self.write_config()
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["status"], "INCOMPLETE", report)
        self.assertIn("target_counts.total has not been specified.", report["missing_materials"])

    def test_missing_joint_declaration_is_incomplete(self):
        self.build()
        del self.config["stratification"]["joint_or_marginal"]
        self.write_config()
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["status"], "INCOMPLETE", report)
        self.assertEqual(report["sampling_process"]["status"], "INCOMPLETE")

    def test_stratification_rejects_wrong_attributes_and_mode(self):
        self.build()
        self.config["stratification"].update(attributes=["weather", "scene", "scene"], joint_or_marginal="unknown")
        self.write_config()
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["status"], "INVALID", report)
        self.assertTrue(any("attributes must contain exactly" in error for error in report["errors"]))
        self.assertTrue(any("joint_or_marginal must be joint or marginal" in error for error in report["errors"]))

    def test_whitespace_declarations_are_incomplete(self):
        self.build()
        self.config["sampling"]["input_order"] = " \t"
        self.config["stratification"].update(allocation_method=" ", missing_attribute_policy="\t")
        self.write_config()
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["status"], "INCOMPLETE", report)
        for field in ("sampling.input_order", "stratification.allocation_method", "stratification.missing_attribute_policy"):
            self.assertTrue(any(field in item for item in report["missing_materials"]))

    def test_partition_digest_uses_canonical_actual_file_hash_mapping(self):
        self.build()
        self.write_config()
        actual_hashes = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest() for name in DATA_PATHS}
        expected = hashlib.sha256(json.dumps(actual_hashes, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        report = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(report["local_partition_sha256"], expected)
        hash_path = self.root / "SHA256SUMS.json"
        hash_path.write_text(json.dumps({"files": dict(reversed(list(actual_hashes.items()))), "algorithm": "sha256"}, indent=4), encoding="utf-8")
        repeated = verify_bundle(self.root, expected_counts=self.counts)
        self.assertEqual(repeated["local_partition_sha256"], expected)

    def test_clean_evaluation_reference_records_hash_and_zero_overlaps(self):
        self.build()
        self.write_config()
        reference = self.root / "evaluation.txt"
        payload = b"eval-a.jpg\neval-b.jpg\n"
        reference.write_bytes(payload)
        report = verify_bundle(self.root, expected_counts=self.counts, evaluation_list=reference)
        self.assertEqual(report["status"], "PASS", report)
        audit = report["evaluation_audit"]
        self.assertEqual(audit["status"], "VERIFIED")
        self.assertEqual(audit["reference_count"], 2)
        self.assertEqual(audit["reference_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(audit["overlap_counts"], {"train": 0, "val": 0, "test": 0})

    def test_evaluation_overlap_with_training_or_model_selection_is_invalid(self):
        self.build()
        self.write_config()
        reference = self.root / "evaluation.txt"
        for split, name in (("train", "EXAMPLE-A.JPG"), ("val", "example-c.jpg")):
            with self.subTest(split=split):
                reference.write_text(name + "\n", encoding="utf-8")
                report = verify_bundle(self.root, expected_counts=self.counts, evaluation_list=reference)
                self.assertEqual(report["status"], "INVALID", report)
                self.assertEqual(report["evaluation_audit"]["status"], "INVALID")
                self.assertEqual(report["evaluation_audit"]["overlap_counts"][split], 1)
                self.assertTrue(any(f"overlaps local {split}" in error for error in report["errors"]))

    def test_local_test_evaluation_overlap_is_reported_without_leakage_failure(self):
        self.build()
        self.write_config()
        reference = self.root / "evaluation.txt"
        reference.write_text("example-d.jpg\n", encoding="utf-8")
        report = verify_bundle(self.root, expected_counts=self.counts, evaluation_list=reference)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["evaluation_audit"]["overlap_filenames"]["test"], ["example-d.jpg"])
        self.assertEqual(report["evaluation_audit"]["overlap_counts"], {"train": 0, "val": 0, "test": 1})
        self.assertTrue(any("overlaps local test" in warning for warning in report["warnings"]))

    def test_evaluation_reference_rejects_duplicates_empty_and_noncanonical_names(self):
        self.build()
        self.write_config()
        reference = self.root / "evaluation.txt"
        for content in ("eval.jpg\nEVAL.JPG\n", "", "\n", "folder/eval.jpg\n", '"eval.jpg"\n', " eval.jpg\n"):
            with self.subTest(content=content):
                reference.write_text(content, encoding="utf-8")
                report = verify_bundle(self.root, expected_counts=self.counts, evaluation_list=reference)
                self.assertEqual(report["status"], "INVALID", report)
                self.assertEqual(report["evaluation_audit"]["status"], "INVALID")
                self.assertIsNone(report["evaluation_audit"]["reference_sha256"])


if __name__ == "__main__":
    unittest.main()
