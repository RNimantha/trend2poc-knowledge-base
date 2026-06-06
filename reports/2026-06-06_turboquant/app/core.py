import numpy as np
from scipy.linalg import hadamard
from typing import Tuple


def next_power_of_two(n: int) -> int:
    """Return the next power of two greater than or equal to n."""
    return 1 << (n - 1).bit_length()


def randomized_hadamard_transform(vectors: np.ndarray) -> np.ndarray:
    """
    Apply randomized Hadamard transform to each vector.
    Pads vectors to next power of two dimension if needed.

    Args:
        vectors: shape (N, D)

    Returns:
        transformed_vectors: shape (N, D_padded)
    """
    n_vectors, dim = vectors.shape
    dim_padded = next_power_of_two(dim)

    # Pad vectors with zeros if needed
    if dim_padded > dim:
        padded_vectors = np.zeros((n_vectors, dim_padded), dtype=vectors.dtype)
        padded_vectors[:, :dim] = vectors
    else:
        padded_vectors = vectors.copy()

    # Generate random sign flips
    signs = np.random.choice([-1, 1], size=(n_vectors, dim_padded))
    signed_vectors = padded_vectors * signs

    # Hadamard matrix is implicit via fast transform
    # Use fast Walsh-Hadamard transform
    transformed_vectors = fast_walsh_hadamard_transform(signed_vectors) / np.sqrt(dim_padded)
    return transformed_vectors


def fast_walsh_hadamard_transform(x: np.ndarray) -> np.ndarray:
    """
    Perform the Fast Walsh-Hadamard Transform (FWHT) on each row of x.

    Args:
        x: shape (N, D) where D is power of two

    Returns:
        transformed: shape (N, D)
    """
    N, D = x.shape
    h = 1
    y = x.copy()
    while h < D:
        for i in range(0, D, h * 2):
            for j in range(i, i + h):
                a = y[:, j]
                b = y[:, j + h]
                y[:, j] = a + b
                y[:, j + h] = a - b
        h *= 2
    return y


def polar_quant(vectors: np.ndarray, num_levels: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    PolarQuant: quantize vectors by their norm and direction.

    Args:
        vectors: shape (N, D)
        num_levels: number of quantization levels for norm

    Returns:
        quantized_vectors: shape (N, D) with quantized values
        norms: original norms of vectors
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    directions = vectors / norms

    # Quantize norms uniformly between min and max norm
    min_norm = norms.min()
    max_norm = norms.max()

    # Avoid division by zero if all norms equal
    if max_norm - min_norm < 1e-12:
        quantized_norms = np.full_like(norms, min_norm)
    else:
        levels = np.linspace(min_norm, max_norm, num_levels)
        indices = np.clip(np.searchsorted(levels, norms, side='right') - 1, 0, num_levels - 1)
        quantized_norms = levels[indices]

    quantized_vectors = directions * quantized_norms
    return quantized_vectors, norms.squeeze()


def qjl_residual_correction(original_vectors: np.ndarray, quantized_vectors: np.ndarray) -> np.ndarray:
    """
    Apply QJL residual correction to reduce bias in inner product estimation.

    Args:
        original_vectors: shape (N, D)
        quantized_vectors: shape (N, D)

    Returns:
        corrected_vectors: shape (N, D)
    """
    residuals = original_vectors - quantized_vectors

    # Generate random projection matrix for Johnson-Lindenstrauss lemma
    N, D = original_vectors.shape
    proj_dim = min(32, D)  # small projection dimension
    random_proj = np.random.normal(0, 1 / np.sqrt(proj_dim), size=(D, proj_dim))

    projected_residuals = residuals @ random_proj
    correction = projected_residuals @ random_proj.T

    corrected_vectors = quantized_vectors + correction
    return corrected_vectors


def reconstruct_vectors(compressed_vectors: np.ndarray, original_dim: int) -> np.ndarray:
    """
    Reconstruct vectors to original dimension by trimming padding.

    Args:
        compressed_vectors: shape (N, D_padded)
        original_dim: int

    Returns:
        reconstructed_vectors: shape (N, original_dim)
    """
    return compressed_vectors[:, :original_dim]


def compute_distortion(original_vectors: np.ndarray, reconstructed_vectors: np.ndarray) -> float:
    """
    Compute average relative mean squared error distortion.

    Args:
        original_vectors: shape (N, D)
        reconstructed_vectors: shape (N, D)

    Returns:
        average_relative_mse: float
    """
    mse = np.mean((original_vectors - reconstructed_vectors) ** 2, axis=1)
    norm_sq = np.mean(np.sum(original_vectors ** 2, axis=1))
    relative_mse = np.mean(mse) / (norm_sq + 1e-12)
    return relative_mse


def compute_inner_product_error(original_vectors: np.ndarray, reconstructed_vectors: np.ndarray) -> float:
    """
    Compute average absolute error in inner product preservation between pairs of vectors.

    Args:
        original_vectors: shape (N, D)
        reconstructed_vectors: shape (N, D)

    Returns:
        average_inner_product_error: float
    """
    N = original_vectors.shape[0]
    # Sample 100 random pairs or all if less
    num_pairs = min(100, N * (N - 1) // 2)
    if num_pairs == 0:
        return 0.0

    rng = np.random.default_rng(seed=42)
    pairs = set()
    while len(pairs) < num_pairs:
        i = rng.integers(0, N)
        j = rng.integers(0, N)
        if i != j:
            pairs.add(tuple(sorted((i, j))))
    pairs = list(pairs)

    errors = []
    for i, j in pairs:
        original_ip = np.dot(original_vectors[i], original_vectors[j])
        reconstructed_ip = np.dot(reconstructed_vectors[i], reconstructed_vectors[j])
        errors.append(abs(original_ip - reconstructed_ip))

    return float(np.mean(errors))
