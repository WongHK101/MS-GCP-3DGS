#!/usr/bin/env python3

import copy
import unittest

from validate_stage0_5_resource_preflight import GIB, validate_preflight


class ResourcePreflightTests(unittest.TestCase):
    def setUp(self):
        self.contract = {
            "schema": "gs_gcp_resource_probe_contract_v2",
            "resource_gates": {
                "host_peak_fraction_of_cgroup_limit": 0.8,
                "host_minimum_headroom_gib": 24,
                "gpu_peak_fraction_of_total": 0.75,
                "gpu_minimum_headroom_gib": 24,
                "fd_peak_max": 4096,
                "fd_stable_absolute_max": 256,
                "fd_stable_baseline_delta_max": 64,
                "fd_last_ten_range_max": 8,
                "process_tree_cgroup_delta_absolute_tolerance_gib": 8,
                "process_tree_cgroup_delta_relative_tolerance": 0.25,
            },
        }
        self.resource = {
            "status": "PASS", "probe_complete": True,
            "cgroup_memory_limit_bytes": 110 * GIB,
            "cgroup_memory_baseline_bytes": 2 * GIB,
            "cgroup_observed_peak_bytes": 12 * GIB,
            "gpu_memory_total_mib_per_device": [96 * 1024],
            "peak_device_memory_used_mib": 12000,
            "peak_gpu_memory_mib": 11000,
            "memory_events_delta": {"oom": 0, "oom_kill": 0, "max": 0},
            "fd_peak": 120, "fd_last_ten_min": 80, "fd_last_ten_max": 82,
            "process_tree_sampled_peak_rss_kib": 10 * GIB // 1024,
        }
        self.camera = {
            "schema": "gs_gcp_original_3dgs_camera_load_preflight_v2",
            "status": "PASS", "resolution": 4, "data_device": "cuda",
            "host_allocator_policy": "glibc_malloc_trim_threshold_zero_v1",
            "malloc_trim_threshold_env": "0",
            "camera_count": 10, "camera_records_read_count": 10,
            "camera_tensors_materialized_count": 10,
            "currently_open_source_image_count": 0,
            "points3d_tracks_read": False, "fd_before": 20, "fd_after": 24,
            "jpeg_fds_after_stabilization": [],
            "theoretical_camera_tensor_bytes": 8 * GIB,
            "actual_camera_tensor_bytes": 8 * GIB,
            "torch_cuda_allocated_before": 0, "torch_cuda_allocated_after": 8 * GIB,
            "torch_cuda_reserved_before": 0, "torch_cuda_reserved_after": 9 * GIB,
            "max_normalized_ray_coordinate_error": 1e-15,
        }

    def test_pass(self):
        self.assertEqual(validate_preflight(self.contract, self.resource, self.camera)["status"], "PASS")

    def test_oom_is_blocker(self):
        resource = copy.deepcopy(self.resource)
        resource["memory_events_delta"]["oom"] = 1
        result = validate_preflight(self.contract, resource, self.camera)
        self.assertEqual(result["status"], "BLOCKER")

    def test_tensor_mismatch_is_blocker(self):
        camera = copy.deepcopy(self.camera)
        camera["actual_camera_tensor_bytes"] += 4
        result = validate_preflight(self.contract, self.resource, camera)
        self.assertEqual(result["status"], "BLOCKER")

    def test_unexplained_cgroup_gap_is_blocker(self):
        resource = copy.deepcopy(self.resource)
        resource["cgroup_observed_peak_bytes"] = 60 * GIB
        resource["process_tree_sampled_peak_rss_kib"] = 10 * GIB // 1024
        result = validate_preflight(self.contract, resource, self.camera)
        self.assertEqual(result["status"], "BLOCKER")

    def test_missing_allocator_policy_is_blocker(self):
        camera = copy.deepcopy(self.camera)
        camera["malloc_trim_threshold_env"] = None
        result = validate_preflight(self.contract, self.resource, camera)
        self.assertEqual(result["status"], "BLOCKER")


if __name__ == "__main__":
    unittest.main()
