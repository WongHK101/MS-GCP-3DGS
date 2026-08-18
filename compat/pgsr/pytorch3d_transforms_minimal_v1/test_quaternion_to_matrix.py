from __future__ import annotations

import math

import torch

from pytorch3d.transforms import quaternion_to_matrix


def test_identity_and_axis_rotations() -> None:
    q = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0],
            [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0],
            [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
        ],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(quaternion_to_matrix(q), expected, atol=1e-12, rtol=1e-12)


def test_scale_invariance_orthogonality_and_determinant() -> None:
    generator = torch.Generator().manual_seed(0)
    q = torch.randn((64, 4), generator=generator, dtype=torch.float64)
    rotation = quaternion_to_matrix(q)
    scaled_rotation = quaternion_to_matrix(q * 7.25)
    identity = torch.eye(3, dtype=torch.float64).expand(64, -1, -1)

    torch.testing.assert_close(rotation, scaled_rotation, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), identity, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(64, dtype=torch.float64), atol=1e-12, rtol=1e-12)
