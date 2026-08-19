#!/usr/bin/env python3
"""CPU-only contract tests for the frozen MetroGS training wrapper."""

from __future__ import annotations

from pathlib import Path

from run_metrogs_training import build_commands, build_resolved_config


def official_config() -> dict:
    return {
        "save_val": False,
        "model": {
            "gaussian": {
                "class_path": "internal.models.metrogs.Gaussian2D",
                "init_args": {"sh_degree": 2},
            },
            "metric": {
                "class_path": "internal.metrics.metrogs_metrics.DistributedMetrics",
                "init_args": {
                    "single_view_from": 0,
                    "multi_view_from": 50_000,
                    "depth_loss_type": "l1+ssim",
                    "depth_loss_weight": {
                        "init": 0.5,
                        "final_factor": 0.005,
                        "max_steps": 50_000,
                    },
                },
            },
            "renderer": {
                "class_path": "internal.renderers.metrogs_renderer.DistributedRenderer",
                "init_args": {
                    "use_app": True,
                    "aabb": [-12, -8, -1, 11, 9, 6],
                },
            },
            "density": {
                "class_path": "internal.density_controllers.metrogs_density_controller.DistributedController",
                "init_args": {"voxel_size": 0.1, "densify_until_iter": 50_000},
            },
        },
        "trainer": {
            "strategy": {"class_path": "internal.mp_strategy.MPStrategy"},
            "devices": [0, 1, 2, 3],
            "max_steps": 150_000,
        },
        "data": {
            "use_multi_view": True,
            "batch_size": 4,
            "path": "data/matrix_city/aerial/train/block_all",
            "parser": {
                "class_path": "internal.dataparsers.estimated_mask_depth_colmap_dataparser.EstimatedDepthColmap",
                "init_args": {
                    "additional_ply_path": "add_ply/mc_aerial.ply",
                    "split_mode": "reconstruction",
                    "down_sample_factor": 1.2,
                },
            },
        },
        "save_iterations": [150_000],
    }


def main() -> int:
    common = {
        "official": official_config(),
        "dataset": Path("/data/gcp_3000_20260602"),
        "additional_ply": Path("/data/gcp_3000_20260602/additional_points/pi3.ply"),
        "model_path": Path("/run/model"),
    }
    formal = build_resolved_config(mode="formal", iterations=150_000, **common)
    assert formal["seed_everything"] == 0
    assert formal["trainer"]["devices"] == [0]
    assert formal["trainer"]["max_steps"] == 150_000
    assert formal["data"]["batch_size"] == 4
    assert formal["data"]["parser"]["init_args"]["down_sample_factor"] == 1.0
    assert formal["model"]["renderer"]["init_args"]["aabb"] is None
    assert formal["model"]["metric"]["init_args"]["multi_view_from"] == 50_000
    assert formal["save_iterations"] == [150_000]
    assert formal["logger"] == "tensorboard"
    assert Path(formal["output"]) == Path("/run") and formal["name"] == "model"

    qualification = build_resolved_config(
        mode="qualification", iterations=8, **common
    )
    assert qualification["trainer"]["max_steps"] == 8
    assert qualification["save_iterations"] == [8]
    assert qualification["model"]["metric"]["init_args"]["single_view_from"] == 0
    assert qualification["model"]["metric"]["init_args"]["multi_view_from"] == 4
    assert qualification["logger"] == "tensorboard"

    train, merge = build_commands(
        python=Path("/env/bin/python"),
        repo=Path("/repo"),
        model_path=Path("/run/model"),
        resolved_config=Path("/run/metrogs_frozen_training_config.yaml"),
    )
    assert Path(train[0]) == Path("/env/bin/python")
    assert train[1:2] == ["-B"]
    assert Path(train[2]) == Path("/repo/main_bsz.py")
    assert train[3:5] == ["fit", "--config"]
    assert Path(train[5]) == Path("/run/metrogs_frozen_training_config.yaml")
    assert Path(merge[-2]) == Path("/repo/utils/merge_distributed_ckpts.py")
    assert Path(merge[-1]) == Path("/run/model")

    for mode, iterations in (("formal", 149_999), ("qualification", 4)):
        try:
            build_resolved_config(mode=mode, iterations=iterations, **common)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid {mode} budget was accepted")
    print("metrogs_training_command_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
