"""Safe metadata and duplicate-copy tests, without Torch or user datasets."""
import io
import json
import pickle
import unittest
import zipfile
from collections import OrderedDict

from scripts.analyze_lookback_results import read_archive
from scripts.audit_ecl_training_archive import (
    MetadataUnpickler, StorageSpec, checkpoint_metadata, json_keys,
    tensor_placeholder,
)


class ECLTrainingArchiveTests(unittest.TestCase):
    def payload(self):
        from tests.test_lookback_results_analysis import LookbackAuditTests
        return LookbackAuditTests().make_record()[1]

    def test_identical_aggregate_copies_are_not_extra_runs(self):
        payload = self.payload()
        raw = json.dumps(payload).encode()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("root/result.json", raw)
            z.writestr("root/copy/result.json", raw)
        buf.seek(0)
        self.assertEqual(len(read_archive(buf)), 1)

    def test_conflicting_aggregate_copies_are_not_deduplicated(self):
        payload = self.payload()
        raw = json.dumps(payload).encode()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("result.json", raw)
            payload["created_at_utc"] = "different"
            z.writestr("copy/result.json", json.dumps(payload))
        buf.seek(0)
        self.assertEqual(len(read_archive(buf)), 2)

    def test_checkpoint_config_integer_keys_match_json_keys(self):
        saved = {"config": {"horizon_overrides": {96: {"lr": 0.0005}}},
                 "n_channels": 321, "val_metrics": {"mse": 0.15},
                 "epoch": 2, "model_state": OrderedDict()}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("checkpoint/data.pkl", pickle.dumps(saved))
        metadata = checkpoint_metadata(buf.getvalue())
        self.assertEqual(metadata["config"]["horizon_overrides"]["96"]["lr"], 0.0005)
        self.assertEqual(metadata["epoch"], 2)

    def test_json_key_collisions_are_rejected(self):
        with self.assertRaises(ValueError):
            json_keys({96: 1, "96": 2})

    def test_foreign_pickle_globals_are_rejected(self):
        serialized_function = pickle.dumps(eval)  # Serialize only; never execute it.
        with self.assertRaises(pickle.UnpicklingError):
            MetadataUnpickler(io.BytesIO(serialized_function)).load()

    def test_model_classes_are_not_imported(self):
        with self.assertRaises(pickle.UnpicklingError):
            MetadataUnpickler(io.BytesIO()).find_class("torch.nn", "Module")

    def test_tensor_placeholder_does_not_allocate_tensor_data(self):
        tensor = tensor_placeholder(StorageSpec(1024, "storage0"), 0, (32, 32), (32, 1))
        self.assertEqual(tensor.shape, (32, 32))
        self.assertEqual(tensor.storage_key, "storage0")


if __name__ == "__main__":
    unittest.main()
