import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # 1. Validate input dimensions
    if x.ndim not in [3, 4]:
        raise ValueError(f"Expected input with 3 or 4 dimensions, got {x.ndim}")

    # 2. Compute the mean over the spatial dimensions (the last two axes)
    # For (C, H, W), axes (1, 2) are H and W.
    # For (N, C, H, W), axes (2, 3) are H and W.
    # Using negative indexing (-2, -1) covers both cases elegantly.
    return np.mean(x, axis=(-2, -1), dtype=np.float64)