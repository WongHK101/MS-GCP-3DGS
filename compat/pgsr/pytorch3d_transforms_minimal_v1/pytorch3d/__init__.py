"""Minimal compatibility namespace required by frozen PGSR.

Only ``pytorch3d.transforms.quaternion_to_matrix`` is provided.  PGSR imports
no other PyTorch3D API in its training or benchmark packet-export path.
"""
