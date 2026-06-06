# turboquant-poc

## Project Description

This project demonstrates the core TurboQuant vector quantization pipeline, which compresses high-dimensional vectors with minimal distortion and unbiased inner product estimation. It implements key steps such as randomized Hadamard transform, PolarQuant, QJL residual correction, and vector reconstruction on synthetic data.

## Prerequisites

- Python 3.9 or higher
- No external API accounts or secrets required

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/turboquant-poc.git
   cd turboquant-poc
   ```

2. Create and activate a Python virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

This POC does not require any environment variables to run. However, a `.env.example` file is provided for future extensibility.

If you want to add environment variables, copy `.env.example` to `.env` and fill in values.

## Running the Demo

Run the main script to load sample vectors, compress and decompress them using TurboQuant, and print distortion and inner product preservation metrics:

```bash
python app/main.py
```

## Example Output

```
Loaded 100 vectors of dimension 128
Compression and decompression completed.
Average relative distortion (MSE): 0.0123
Average inner product preservation error: 0.0087
```

## How This POC Demonstrates the Concept

- Implements a simplified TurboQuant pipeline on synthetic high-dimensional vectors.
- Applies randomized Hadamard transform to spread vector energy.
- Uses PolarQuant for quantization preserving vector norms.
- Applies QJL residual correction to reduce bias in inner product estimation.
- Measures distortion and inner product preservation to validate compression quality.

## Known Limitations

- Does not implement full TurboQuant optimizations or GPU acceleration.
- Uses synthetic data instead of real KV cache vectors.
- Simplifies PolarQuant and QJL steps for educational clarity.
- No online streaming interface; processes batch input from file.
- No integration with actual LLM inference pipelines.
