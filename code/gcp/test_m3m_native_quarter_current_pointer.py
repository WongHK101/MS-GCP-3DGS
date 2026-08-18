from __future__ import annotations

import json
from pathlib import Path

from m3m_native_quarter_protocol import PROTOCOL_ID


REPO_ROOT = Path(__file__).resolve().parents[2]
POINTER_PATH = REPO_ROOT / "configs" / "m3m_gcp_native_quarter_current.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_pointer_selects_one_consistent_protocol() -> None:
    pointer = load_json(POINTER_PATH)
    assert pointer["schema"] == "m3m_gcp_native_quarter_current_pointer_v1"
    assert pointer["status"] == "ACTIVE"
    assert pointer["protocol_id"] == PROTOCOL_ID

    document_path = REPO_ROOT / pointer["protocol_document"]
    release_pin_path = REPO_ROOT / pointer["protocol_release_pin"]
    registry_path = REPO_ROOT / pointer["method_registry"]
    batch_plan_path = REPO_ROOT / pointer["batch_execution_plan"]
    registry_validation_path = REPO_ROOT / pointer["method_registry_validation"]
    assert document_path.is_file()
    assert release_pin_path.is_file()
    assert registry_path.is_file()
    assert batch_plan_path.is_file()
    assert registry_validation_path.is_file()

    release_pin = load_json(release_pin_path)
    registry = load_json(registry_path)
    assert release_pin["protocol_id"] == PROTOCOL_ID
    assert registry["protocol_id"] == PROTOCOL_ID
    assert registry["schema"] == "m3m_gcp_native_quarter_method_registry_v3"
    assert registry["execution_plan"]["path"] == pointer["batch_execution_plan"]
    validation = load_json(registry_validation_path)
    assert validation["schema"] == "m3m_gcp_native_quarter_method_registry_validation_v3"
    assert validation["passed"] is True
    assert validation["batch_id"] == registry["batch_id"]
    assert (
        pointer["source_data_release"]["directory_name"]
        == release_pin["source_data_directory_name"]
        == registry["source_data_release"]["directory_name"]
    )
    assert (
        pointer["source_data_release"]["release_root_digest_sha256"]
        == release_pin["source_data_release_root_digest_sha256"]
        == registry["source_data_release"]["release_root_digest_sha256"]
    )

    document = document_path.read_text(encoding="utf-8")
    assert PROTOCOL_ID in document
    assert "状态：**ACTIVE" in document


def test_superseded_protocol_cannot_reenter_active_namespace() -> None:
    pointer = load_json(POINTER_PATH)
    for relative_path in pointer["removed_repository_assets"]:
        assert not (REPO_ROOT / relative_path).exists(), relative_path
    for relative_path in pointer["tombstoned_legacy_entrypoints"]:
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "SUPERSEDED_DO_NOT_USE" in text, relative_path
    for relative_path in pointer["current_batch_execution_components"]:
        assert (REPO_ROOT / relative_path).is_file(), relative_path
    for relative_path in pointer["current_repository_components_with_independent_v1_suffixes"]:
        assert (REPO_ROOT / relative_path).is_file(), relative_path

    forbidden_tokens = (
        "m3m_gcp_native_quarter_geometry_v1",
        "M3M-GCP-native-quarter-benchmark-protocol-v1",
    )
    active_files = [
        *sorted((REPO_ROOT / "configs").glob("m3m_gcp_native_quarter_*")),
        *sorted((REPO_ROOT / "docs").glob("GS_GCP_NATIVE_QUARTER_*")),
        *sorted((REPO_ROOT / "docs" / "protocol_evidence").glob("*native_quarter*")),
    ]
    for path in active_files:
        if path == POINTER_PATH or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, f"{token} leaked into {path.relative_to(REPO_ROOT)}"


def test_every_registered_formal_report_is_owned_by_the_current_pointer() -> None:
    pointer = load_json(POINTER_PATH)
    registry = load_json(REPO_ROOT / pointer["method_registry"])
    current_components = {
        *pointer["current_batch_execution_components"],
        *pointer["current_repository_components_with_independent_v1_suffixes"],
    }
    for method in registry["methods"]:
        qualification_report = method.get("qualification_report")
        if qualification_report is not None:
            assert qualification_report in current_components, (
                f"{method['method_id']}: registered qualification report is not owned by "
                "the current protocol pointer"
            )
        report = method.get("formal_3k_result", {}).get("report")
        if report is not None:
            assert report in current_components, (
                f"{method['method_id']}: registered formal report is not owned by "
                "the current protocol pointer"
            )
    gate = registry.get("current_one_use_launch_gate")
    if gate is not None:
        assert gate["path"] in current_components, (
            "current one-use launch gate is not owned by the current protocol pointer"
        )
