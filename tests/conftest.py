import os
import sys

import pytest

# Tests import both `centroid_erasure` and the top-level `main` module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_collection_modifyitems(config, items):
    """Belt and braces: skip GPU-marked tests at collection when no CUDA device
    is present, so they cannot fail during fixture setup on a CPU-only box."""
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if has_cuda:
        return
    skip = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
