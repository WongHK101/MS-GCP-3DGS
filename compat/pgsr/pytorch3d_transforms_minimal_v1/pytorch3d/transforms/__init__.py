"""Semantics-preserving quaternion conversion used by frozen PGSR."""

from __future__ import annotations

import torch


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert real-first quaternions to rotation matrices.

    The formula matches the public PyTorch3D API used by PGSR and supports
    arbitrary leading batch dimensions.  Non-unit quaternions are normalized
    algebraically through the squared-norm denominator.
    """

    real, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    matrix = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * real),
            two_s * (i * k + j * real),
            two_s * (i * j + k * real),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * real),
            two_s * (i * k - j * real),
            two_s * (j * k + i * real),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return matrix.reshape(quaternions.shape[:-1] + (3, 3))


__all__ = ["quaternion_to_matrix"]
