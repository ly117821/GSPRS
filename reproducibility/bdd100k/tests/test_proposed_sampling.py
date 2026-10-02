import sys
import copy
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sample_proposed_split import apportion, select_splits, validate_protocol_config


class ProposedSamplingTests(unittest.TestCase):
    def records(self):
        return [{"name": "synthetic_test_%03d.jpg" % i, "attributes": {"weather": "clear" if i % 3 else "rainy", "timeofday": "daytime", "scene": "city street"}} for i in range(80)]

    def select(self, records, seed=20261001):
        return select_splits(records, protocol="gsprs-bdd100k-proposed-v1", sample_seed=seed, partition_seed=20261002, counts={"train": 35, "val": 10, "test": 5})

    def test_exact_counts_and_no_overlap(self):
        outputs, strata = self.select(self.records())
        self.assertEqual({key: len(value) for key, value in outputs.items()}, {"train": 35, "val": 10, "test": 5})
        self.assertEqual(len(set(sum(outputs.values(), []))), 50)
        self.assertEqual(sum(row["selected_count"] for row in strata), 50)

    def test_input_order_independent(self):
        self.assertEqual(self.select(self.records()), self.select(list(reversed(self.records()))))

    def test_changing_sampling_seed_changes_selection(self):
        self.assertNotEqual(self.select(self.records())[0], self.select(self.records(), seed=20261003)[0])

    def test_duplicate_and_insufficient_pool_rejected(self):
        records = self.records()
        with self.assertRaises(ValueError):
            self.select(records + [records[0]])
        with self.assertRaises(ValueError):
            self.select(records[:20])

    def test_hamilton_tie_breaking_and_capacity(self):
        self.assertEqual(apportion({("b",): 1, ("a",): 1, ("c",): 1}, 2), {("a",): 1, ("b",): 1, ("c",): 0})
        for target in range(9):
            result = apportion({("a",): 1, ("b",): 7}, target)
            self.assertEqual(sum(result.values()), target)
            self.assertLessEqual(result[("a",)], 1)
            self.assertLessEqual(result[("b",)], 7)

    def test_missing_attributes_are_separate_stratum(self):
        records = self.records()
        records[0]["attributes"] = {}
        _, strata = self.select(records)
        self.assertTrue(any(row["weather"] == "" and row["source_count"] == 1 for row in strata))

    def protocol_config(self):
        return json.loads((Path(__file__).resolve().parents[1] / "configs/sampling_config.json").read_text(encoding="utf-8"))

    def test_actual_protocol_and_seed_zero_are_accepted(self):
        config = self.protocol_config()
        config["sampling"]["seed"] = 0
        validate_protocol_config(config)

    def test_false_algorithm_declarations_are_rejected(self):
        config = self.protocol_config()
        for key, value in (("joint_or_marginal", "marginal"), ("allocation_method", "uniform"), ("attributes", ["scene"]), ("allow_missing_attribute_strata", False)):
            changed = copy.deepcopy(config)
            changed["stratification"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol_config(changed)
        changed = copy.deepcopy(config)
        changed["hash_payload"] = "different ranking"
        with self.assertRaises(ValueError):
            validate_protocol_config(changed)

    def test_boolean_seed_and_count_are_rejected(self):
        config = self.protocol_config()
        config["partitioning"]["seed"] = True
        with self.assertRaises(ValueError):
            validate_protocol_config(config)
        config = self.protocol_config()
        config["target_counts"]["total"] = True
        with self.assertRaises(ValueError):
            validate_protocol_config(config)


if __name__ == "__main__":
    unittest.main()
