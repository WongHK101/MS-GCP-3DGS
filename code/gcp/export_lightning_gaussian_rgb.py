#!/usr/bin/env python3
"""Render CityGaussianV2 or MetroGS heldout RGB from a Lightning checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_citygaussian_v2_depth_maps import (  # noqa: E402
    build_frozen_cameras,
    load_citygaussian_runtime,
    resolve_sparse_model,
)
from rgb_quality_contract import (  # noqa: E402
    RgbRenderWriter,
    git_identity,
    role_rows,
    sha256_file,
)


EXPECTED_CLASSES = {
    "citygaussian_v2": "SepDepthTrim2DGSRenderer",
    "metrogs": "DistributedRendererImpl",
}


def _map_metrogs_appearance_ids(
    *,
    runtime: dict[str, Any],
    sparse_model: Path,
    writer: RgbRenderWriter,
    training_cameras_json: Path,
    renderer: Any,
) -> tuple[list[tuple[str, Any, int]], dict[str, dict[str, Any]]]:
    from internal.renderers.metrogs_renderer import find_most_similar_cameras  # noqa: WPS433

    training_payload = json.loads(training_cameras_json.read_text(encoding="utf-8"))
    if not isinstance(training_payload, list):
        raise ValueError("MetroGS cameras.json must be a list")
    train_role_names = {str(row["image_name"]) for row in role_rows(writer.input_manifest, "train")}
    ordered_train_names = [Path(str(row["img_name"])).name for row in training_payload]
    if len(ordered_train_names) != len(train_role_names) or set(ordered_train_names) != train_role_names:
        raise ValueError("MetroGS training cameras.json does not exactly match the frozen train role")
    n_appearances = int(getattr(renderer, "n_appearances", -1))
    if n_appearances != len(ordered_train_names):
        raise ValueError(
            f"MetroGS checkpoint appearance count mismatch: {getattr(renderer, 'n_appearances', None)}"
        )
    # MetroGS writes cameras.json before renderer.setup().  The JSON
    # appearance_id therefore still reflects the COLMAP camera group (all zero
    # in this scene).  Its official setup then assigns camera.idx to each
    # training appearance.  cameras.json `id` is that frozen dataset index.
    appearance_ids = [int(row["id"]) for row in training_payload]
    normalized_appearance_ids = [
        float(row["normalized_appearance_id"]) for row in training_payload
    ]
    if sorted(appearance_ids) != list(range(n_appearances)):
        raise ValueError("MetroGS cameras.json appearance IDs are not a complete unique range")
    if not all(0.0 <= value <= 1.0 for value in normalized_appearance_ids):
        raise ValueError("MetroGS cameras.json normalized appearance ID is outside [0,1]")

    train_tuples = build_frozen_cameras(runtime, sparse_model, ordered_train_names)
    for index, (_name, camera, _image_id) in enumerate(train_tuples):
        camera.appearance_id.copy_(camera.appearance_id.new_tensor(appearance_ids[index]))
        camera.normalized_appearance_id.copy_(
            camera.normalized_appearance_id.new_tensor(normalized_appearance_ids[index])
        )
    test_tuples = build_frozen_cameras(runtime, sparse_model, writer.expected_names)
    matches = find_most_similar_cameras(
        [camera for _name, camera, _image_id in train_tuples],
        [camera for _name, camera, _image_id in test_tuples],
        alpha=0.7,
    )
    if len(matches) != len(test_tuples):
        raise RuntimeError("MetroGS appearance matching returned incomplete coverage")
    records: dict[str, dict[str, Any]] = {}
    for test_index, train_index, distance in matches:
        if not (0 <= int(test_index) < len(test_tuples)) or not (
            0 <= int(train_index) < len(train_tuples)
        ):
            raise RuntimeError("MetroGS appearance matcher returned an invalid index")
        test_name, test_camera, _test_image_id = test_tuples[int(test_index)]
        train_name = train_tuples[int(train_index)][0]
        appearance_id = appearance_ids[int(train_index)]
        test_camera.appearance_id.copy_(test_camera.appearance_id.new_tensor(appearance_id))
        records[test_name] = {
            "appearance_policy": "official_nearest_training_camera_geometry_alpha_0_7_v1",
            "appearance_id": appearance_id,
            "normalized_appearance_id": float(test_camera.normalized_appearance_id.item()),
            "matched_training_camera_json_index": int(train_index),
            "matched_training_image_name": train_name,
            "combined_pose_distance": float(distance),
        }
    return test_tuples, records


def export(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    camera_root = args.camera_root.expanduser().resolve()
    sparse_model = resolve_sparse_model(camera_root)
    if args.method_id not in EXPECTED_CLASSES:
        raise ValueError(f"unsupported Lightning Gaussian method: {args.method_id}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    writer = RgbRenderWriter(
        contract_path=args.rgb_contract,
        input_manifest_path=args.input_manifest,
        scene=args.scene,
        method_id=args.method_id,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
    )
    runtime = load_citygaussian_runtime(repo)
    torch = runtime["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("formal Lightning Gaussian RGB export requires CUDA")

    old_cwd = Path.cwd()
    os.chdir(repo)
    checkpoint_payload = None
    try:
        model, renderer, checkpoint_payload = runtime[
            "GaussianModelLoader"
        ].initialize_model_and_renderer_from_checkpoint_file(
            str(checkpoint), device="cuda", eval_mode=True, pre_activate=True
        )
        expected_class = EXPECTED_CLASSES[args.method_id]
        if renderer.__class__.__name__ != expected_class:
            raise RuntimeError(
                f"expected {expected_class}, got {renderer.__class__.__module__}."
                f"{renderer.__class__.__name__}"
            )
        appearance_records: dict[str, dict[str, Any]] = {}
        if args.method_id == "metrogs":
            if args.training_cameras_json is None:
                raise ValueError("MetroGS requires --training_cameras_json")
            frozen_cameras, appearance_records = _map_metrogs_appearance_ids(
                runtime=runtime,
                sparse_model=sparse_model,
                writer=writer,
                training_cameras_json=args.training_cameras_json.expanduser().resolve(),
                renderer=renderer,
            )
        else:
            if args.training_cameras_json is not None:
                raise ValueError("--training_cameras_json is MetroGS-only")
            frozen_cameras = build_frozen_cameras(runtime, sparse_model, writer.expected_names)

        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            for image_name, camera, source_image_id in frozen_cameras:
                camera = camera.to_device("cuda")
                payload = renderer(camera, model, background)
                if not isinstance(payload, dict) or "render" not in payload:
                    raise RuntimeError(f"renderer did not return RGB for {image_name}")
                camera_record = {
                    "source_colmap_image_id": int(source_image_id),
                    "appearance_policy": args.appearance_policy,
                    **appearance_records.get(image_name, {}),
                }
                writer.save(image_name, payload["render"], camera_record=camera_record)
    finally:
        if checkpoint_payload is not None:
            del checkpoint_payload
        os.chdir(old_cwd)

    renderer_source = (
        repo / "internal" / "renderers" / "metrogs_renderer.py"
        if args.method_id == "metrogs"
        else repo / "internal" / "renderers" / "sep_depth_trim_2dgs_renderer.py"
    )
    provenance = {
        "adapter_kind": "lightning_gaussian_rgb_v1",
        "adapter_path": str(Path(__file__).resolve()),
        "adapter_sha256": sha256_file(Path(__file__).resolve()),
        "renderer_repository": git_identity(repo),
        "renderer_class": EXPECTED_CLASSES[args.method_id],
        "renderer_source_path": str(renderer_source),
        "renderer_source_sha256": sha256_file(renderer_source),
        "formal_model_path": str(checkpoint),
        "formal_model_sha256": sha256_file(checkpoint),
        "iteration": int(args.iteration),
        "camera_source_root": str(camera_root),
        "frozen_sparse_model": str(sparse_model),
        "frozen_sparse_model_sha256": {
            name: sha256_file(sparse_model / name)
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        },
        "white_background": False,
        "appearance_policy": args.appearance_policy,
        "training_cameras_json": (
            {
                "path": str(args.training_cameras_json.expanduser().resolve()),
                "sha256": sha256_file(args.training_cameras_json.expanduser().resolve()),
            }
            if args.training_cameras_json is not None
            else None
        ),
        "heldout_rgb_used_by_adapter": False,
        "test_time_optimization": False,
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "device_name": torch.cuda.get_device_name(0),
        },
    }
    return writer.finalize(provenance=provenance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--camera_root", type=Path, required=True)
    parser.add_argument("--rgb_contract", type=Path, required=True)
    parser.add_argument("--input_manifest", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--method_id", choices=sorted(EXPECTED_CLASSES), required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--manifest_path", type=Path)
    parser.add_argument("--appearance_policy", required=True)
    parser.add_argument("--training_cameras_json", type=Path)
    return parser


def main() -> int:
    manifest = export(build_parser().parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
