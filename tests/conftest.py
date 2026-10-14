import numpy as np
import pytest


def clustered(n, dim=512, clusters=1000, noise=1.0, seed=0):
    """Unit vectors = random cluster center + noise (as run_benchmark.c in genesis-live-core)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((clusters, dim)).astype(np.float32)
    x = centers[rng.integers(0, clusters, n)] + noise * rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "l0.db")
