import pytest
import numpy as np
from app.core import (
    randomized_hadamard_transform,
    polar_quant,
    qjl_residual_correction,
    reconstruct_vectors,
    compute_distortion,
    compute_inner_product_error,
)
from app.config import settings


def test_config_loads() -> None:
    # Config should load without error
    assert settings is not None


def test_turboquant_pipeline_smoke() -> None:
    # Generate synthetic data
    np.random.seed(0)
    vectors = np.random.randn(50, 64)

    transformed = randomized_hadamard_transform(vectors)
    quantized, norms = polar_quant(transformed, num_levels=16)
    corrected = qjl_residual_correction(transformed, quantized)
    reconstructed = reconstruct_vectors(corrected, original_dim=64)

    distortion = compute_distortion(vectors, reconstructed)
    inner_product_err = compute_inner_product_error(vectors, reconstructed)

    # Basic assertions
    assert transformed.shape[0] == vectors.shape[0]
    assert quantized.shape == transformed.shape
    assert corrected.shape == transformed.shape
    assert reconstructed.shape == vectors.shape

    # Distortion should be a small positive number
    assert 0 <= distortion < 0.1

    # Inner product error should be small
    assert 0 <= inner_product_err < 0.1
