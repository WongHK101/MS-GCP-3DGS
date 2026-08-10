import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from verify_m3m_native_quarter_geometry_outputs import verify  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class M3MGeometryOutputVerifierTest(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        eval_dir = root / "eval"
        eval_dir.mkdir()
        dependencies = root / "dependencies"
        dependencies.mkdir()
        protocol_manifest = dependencies / "protocol_release_manifest.json"
        data_contract = dependencies / "DATA_CONTRACT_DRAFT.json"
        packet_manifest = dependencies / "depth_export_manifest.json"
        for path, payload in (
            (protocol_manifest, {"kind": "protocol"}),
            (data_contract, {"kind": "data"}),
            (packet_manifest, {"kind": "packet"}),
        ):
            write_json(path, payload)
        sim3_path = dependencies / "common_sim3.json"
        write_json(sim3_path, {
            "schema": "m3m_gcp_native_quarter_common_sim3_v2",
            "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
            "scene": "synthetic_scene",
            "method_result_refit_forbidden": True,
            "transform": {
                "scale": 1.0,
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "translation": [0.0, 0.0, 0.0],
            },
        })

        observation_fields = [
            "observation_id", "scene", "point_name", "role", "image_name", "u_px", "v_px",
            "view_class", "azimuth_bin_45deg", "valid", "failure_reason",
        ]
        observations = []
        for point_name, role in (("C1", "control"), ("K1", "checkpoint")):
            for index, (view_class, azimuth_bin) in enumerate((("nadir", 0), ("nadir", 1), ("oblique", 0), ("oblique", 2))):
                observations.append({
                    "observation_id": f"{point_name}-{index}",
                    "scene": "synthetic_scene",
                    "point_name": point_name,
                    "role": role,
                    "image_name": f"{point_name}-{index}.JPG",
                    "u_px": 1.5,
                    "v_px": 2.5,
                    "view_class": view_class,
                    "azimuth_bin_45deg": azimuth_bin,
                    "valid": True,
                    "failure_reason": "",
                })
        write_csv(eval_dir / "observation_samples.csv", observation_fields, observations)

        point_fields = [
            "scene", "point_name", "role", "surface_level", "passed", "failure_reasons",
            "expected_observation_count", "required_valid_observation_count", "valid_observation_count",
            "valid_nadir_count", "valid_oblique_count", "valid_oblique_azimuth_bin_count",
            "valid_oblique_azimuth_bins_45deg", "max_oblique_azimuth_circular_bin_separation",
            "required_oblique_azimuth_bin_count", "required_oblique_azimuth_circular_bin_separation",
            "aggregation_group_count", "model_x", "model_y", "model_z", "predicted_e_m",
            "predicted_n_m", "predicted_z_m", "target_e_m", "target_n_m", "target_z_m",
            "residual_e_m", "residual_n_m", "residual_z_m", "error_h_m", "error_z_m", "error_3d_m",
            "multiview_scatter_median_m", "multiview_scatter_p90_m", "multiview_scatter_max_m",
        ]
        points = []
        for point_name, role, xyz in (("C1", "control", (1.0, 2.0, 3.0)), ("K1", "checkpoint", (4.0, 5.0, 6.0))):
            points.append({
                "scene": "synthetic_scene", "point_name": point_name, "role": role,
                "surface_level": "ground", "passed": True, "failure_reasons": "",
                "expected_observation_count": 4, "required_valid_observation_count": 4,
                "valid_observation_count": 4, "valid_nadir_count": 2, "valid_oblique_count": 2,
                "valid_oblique_azimuth_bin_count": 2, "valid_oblique_azimuth_bins_45deg": "[0, 2]",
                "max_oblique_azimuth_circular_bin_separation": 2,
                "required_oblique_azimuth_bin_count": 2,
                "required_oblique_azimuth_circular_bin_separation": 2,
                "aggregation_group_count": 4,
                "model_x": xyz[0], "model_y": xyz[1], "model_z": xyz[2],
                "predicted_e_m": xyz[0], "predicted_n_m": xyz[1], "predicted_z_m": xyz[2],
                "target_e_m": xyz[0], "target_n_m": xyz[1], "target_z_m": xyz[2],
                "residual_e_m": 0.0, "residual_n_m": 0.0, "residual_z_m": 0.0,
                "error_h_m": 0.0, "error_z_m": 0.0, "error_3d_m": 0.0,
                "multiview_scatter_median_m": 0.0, "multiview_scatter_p90_m": 0.0,
                "multiview_scatter_max_m": 0.0,
            })
        write_csv(eval_dir / "point_results.csv", point_fields, points)
        zero_stats = {
            "count": 1, "rmse_h_m": 0.0, "rmse_z_m": 0.0, "rmse_3d_m": 0.0,
            "median_3d_m": 0.0, "p95_3d_m": 0.0, "max_3d_m": 0.0,
        }
        all_stats = dict(zero_stats, count=2)
        summary = {
            "schema": "m3m_gcp_native_quarter_method_evaluation_v2",
            "protocol_id": "m3m_gcp_native_quarter_geometry_v2",
            "scene": "synthetic_scene", "method_id": "synthetic_method",
            "status": "COMPLETE_RANKED", "ranking_eligible": True,
            "physical_surface_claim": False, "method_specific_sim3_fitted": False,
            "common_sim3_path": str(sim3_path), "common_sim3_sha256": sha256_file(sim3_path),
            "packet_manifest": str(packet_manifest), "packet_manifest_sha256": sha256_file(packet_manifest),
            "point_counts": {"control_total": 1, "control_passed": 1, "checkpoint_total": 1, "checkpoint_passed": 1},
            "checkpoint_coverage_rate": 1.0,
            "residual_statistics": {"control": zero_stats, "checkpoint": zero_stats, "all": all_stats},
            "observation_failure_counts": {},
        }
        write_json(eval_dir / "evaluation_summary.json", summary)
        manifest = {
            "schema": "m3m_gcp_native_quarter_evaluator_run_manifest_v2",
            "protocol_release_manifest": str(protocol_manifest),
            "protocol_release_manifest_sha256": sha256_file(protocol_manifest),
            "source_data_contract": str(data_contract),
            "source_data_contract_sha256": sha256_file(data_contract),
            "packet_manifest": str(packet_manifest),
            "packet_manifest_sha256": sha256_file(packet_manifest),
            "operator": "bilinear_raw_moment_ratio_v1",
            "ranking_policy": "complete_checkpoint_coverage_only_v1",
            "sim3_policy": "frozen_common_transform_no_method_refit",
            "outputs": {
                name: sha256_file(eval_dir / name)
                for name in ("observation_samples.csv", "point_results.csv", "evaluation_summary.json")
            },
        }
        write_json(eval_dir / "evaluator_manifest.json", manifest)
        return eval_dir

    def test_verifies_complete_ranked_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = verify(self._fixture(Path(directory)))
            self.assertTrue(result["passed"], result["errors"])
            self.assertEqual(result["point_counts"]["checkpoint_passed"], 1)

    def test_rejects_corrupted_point_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            eval_dir = self._fixture(Path(directory))
            text = (eval_dir / "point_results.csv").read_text(encoding="utf-8")
            (eval_dir / "point_results.csv").write_text(text.replace(",0.0,0.0,0.0,0.0,0.0,0.0,", ",0.0,0.0,0.0,0.0,0.0,1.0,", 1), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify(eval_dir)


if __name__ == "__main__":
    unittest.main()
