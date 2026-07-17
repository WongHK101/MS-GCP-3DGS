from __future__ import annotations

import numpy as np

from verify_gaussian_gcp_eval_outputs import fit_sim3, residual_stats


def main() -> None:
    source = np.asarray([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float64)
    angle = 0.2
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    scale = 1.25
    translation = np.asarray([10.0, -4.0, 2.0], dtype=np.float64)
    target = (scale * (rotation @ source.T)).T + translation
    actual_scale, actual_rotation, actual_translation = fit_sim3(source, target)
    assert abs(actual_scale - scale) <= 1e-12
    assert np.max(np.abs(actual_rotation - rotation)) <= 1e-12
    assert np.max(np.abs(actual_translation - translation)) <= 1e-12

    rows = [
        {"error_h_m": "3", "error_z_m": "4", "error_3d_m": "5"},
        {"error_h_m": "0", "error_z_m": "0", "error_3d_m": "0"},
    ]
    stats = residual_stats(rows)
    assert abs(float(stats["rmse_h_m"]) - np.sqrt(4.5)) <= 1e-12
    assert abs(float(stats["rmse_z_m"]) - np.sqrt(8.0)) <= 1e-12
    assert abs(float(stats["rmse_3d_m"]) - np.sqrt(12.5)) <= 1e-12
    print("PASS test_known_sim3_recovery")
    print("PASS test_residual_statistics")
    print("PASS 2/2")


if __name__ == "__main__":
    main()
