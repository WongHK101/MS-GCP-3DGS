"""Minimal compatibility namespace required by frozen GSPrior.

Only ``pytorch3d.transforms.quaternion_to_matrix`` is provided. GSPrior does
not import another PyTorch3D API in its training or packet-export path.
"""
