import json
import sys
from pathlib import Path
from typing import Any

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


def load_vectors_from_json(file_path: Path) -> np.ndarray:
    """
    Load vectors from a JSON file containing a list of lists.

    Args:
        file_path: Path to JSON file

    Returns:
        vectors: np.ndarray shape (N, D)
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        vectors = np.array(data, dtype=np.float64)
        if vectors.ndim != 2:
            raise ValueError("Input JSON must be a 2D list of vectors.")
        return vectors
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading vectors: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    sample_input_path = Path(__file__).parent.parent / "examples" / "sample_input.json"

    print(f"Loading vectors from {sample_input_path}")
    vectors = load_vectors_from_json(sample_input_path)
    n_vectors, dim = vectors.shape
    print(f"Loaded {n_vectors} vectors of dimension {dim}")

    # Step 1: Randomized Hadamard Transform
    transformed_vectors = randomized_hadamard_transform(vectors)

    # Step 2: PolarQuant quantization
    quantized_vectors, original_norms = polar_quant(transformed_vectors, num_levels=16)

    # Step 3: QJL residual correction
    corrected_vectors = qjl_residual_correction(transformed_vectors, quantized_vectors)

    # Step 4: Reconstruct to original dimension
    reconstructed_vectors = reconstruct_vectors(corrected_vectors, original_dim=dim)

    # Step 5: Compute distortion and inner product preservation
    distortion = compute_distortion(vectors, reconstructed_vectors)
    inner_product_error = compute_inner_product_error(vectors, reconstructed_vectors)

    print("Compression and decompression completed.")
    print(f"Average relative distortion (MSE): {distortion:.6f}")
    print(f"Average inner product preservation error: {inner_product_error:.6f}")


if __name__ == "__main__":
    main()
