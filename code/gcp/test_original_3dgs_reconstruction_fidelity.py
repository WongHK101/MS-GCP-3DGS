import json
import tempfile
import unittest
from pathlib import Path

from original_3dgs_reconstruction_fidelity import merge_chunks


class ReconstructionFidelityMergeTests(unittest.TestCase):
    def test_merges_official_per_view_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunks = root / "chunks"
            chunks.mkdir()
            manifest_rows = []
            values = {
                "chunk_0000": {"00000.png": 0.1, "00001.png": 0.2},
                "chunk_0001": {"00002.png": 0.3},
            }
            for chunk_id, psnr_values in values.items():
                model = chunks / chunk_id
                model.mkdir()
                per_view = {
                    "SSIM": {name: value + 0.4 for name, value in psnr_values.items()},
                    "PSNR": {name: value + 20.0 for name, value in psnr_values.items()},
                    "LPIPS": {name: value for name, value in psnr_values.items()},
                }
                aggregate = {
                    metric: sum(metric_values.values()) / len(metric_values)
                    for metric, metric_values in per_view.items()
                }
                (model / "results.json").write_text(
                    json.dumps({str(model): {"ours_30000": aggregate}}), encoding="utf-8"
                )
                (model / "per_view.json").write_text(
                    json.dumps({str(model): {"ours_30000": per_view}}), encoding="utf-8"
                )
                manifest_rows.append(
                    {
                        "chunk_id": chunk_id,
                        "model_path": str(model),
                        "view_count": len(psnr_values),
                        "image_names": sorted(psnr_values),
                    }
                )
            (chunks / "chunk_manifest.json").write_text(
                json.dumps(
                    {
                        "expected_view_count": 3,
                        "method_name": "ours_30000",
                        "chunks": manifest_rows,
                    }
                ),
                encoding="utf-8",
            )

            summary = merge_chunks(chunks, root / "merged", 3, "ours_30000")
            self.assertEqual(summary["view_count"], 3)
            self.assertAlmostEqual(summary["metrics"]["PSNR"], 20.2)
            self.assertAlmostEqual(summary["metrics"]["SSIM"], 0.6)
            self.assertAlmostEqual(summary["metrics"]["LPIPS"], 0.2)

    def test_rejects_duplicate_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunks = root / "chunks"
            chunks.mkdir()
            rows = []
            for chunk_id in ("chunk_0000", "chunk_0001"):
                model = chunks / chunk_id
                model.mkdir()
                per_view = {metric: {"00000.png": 1.0} for metric in ("SSIM", "PSNR", "LPIPS")}
                aggregate = {metric: 1.0 for metric in per_view}
                (model / "results.json").write_text(
                    json.dumps({str(model): {"ours_30000": aggregate}}), encoding="utf-8"
                )
                (model / "per_view.json").write_text(
                    json.dumps({str(model): {"ours_30000": per_view}}), encoding="utf-8"
                )
                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "model_path": str(model),
                        "view_count": 1,
                        "image_names": ["00000.png"],
                    }
                )
            (chunks / "chunk_manifest.json").write_text(
                json.dumps({"expected_view_count": 2, "method_name": "ours_30000", "chunks": rows}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate per-view metric"):
                merge_chunks(chunks, root / "merged", 2, "ours_30000")


if __name__ == "__main__":
    unittest.main()
