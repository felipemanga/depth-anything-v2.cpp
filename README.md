# depth-anything-v2.cpp

C++ port of [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) for fast, self-contained monocular depth estimation. Built on [ggml](https://github.com/ggml-org/ggml) with GPU acceleration via CUDA, Metal, and Vulkan backends.

## Features

- **Zero dependencies** at runtime — single binary, no Python, no PyTorch
- **GPU acceleration** — CUDA (NVIDIA), Metal (Apple), Vulkan (cross-platform)
- **GGUF model format** — efficient loading, no framework overhead
- **Numerically accurate** — depth output matches Python reference (Corr > 0.999)
- **Multiple architectures** — ViT-Small, ViT-Base, ViT-Large, ViT-Giant2

## Quick Start

### Build

```bash
git clone --recursive https://github.com/felipemanga/depth-anything-v2.cpp.git
cd depth-anything-v2.cpp

# CPU only
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# With CUDA
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDA_CUDA=ON
cmake --build build -j
```

### Convert a Model

```bash
pip install torch numpy gguf

python convert_to_gguf.py \
  ../Depth-Anything-V2/depth_anything_v2_vitb.pth \
  depth_anything_v2_vitb.gguf
```

### Run

```bash
# Predict depth from an image
./build/examples/cli/da_cli predict depth_anything_v2_vitb.gguf input.jpg -o depth.png

# Show model info
./build/examples/cli/da_cli info depth_anything_v2_vitb.gguf

# Benchmark
./build/examples/cli/da_cli bench depth_anything_v2_vitb.gguf input.jpg -n 10
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `predict <model> <image>` | Generate depth map |
| `info <model>` | Print model architecture |
| `bench <model> <image>` | Benchmark inference speed |

### Predict Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o <path>` | `depth_colored.png` | Colored depth output |
| `-d <path>` | `depth_raw.png` | Raw 16-bit depth output |
| `-t <n>` | auto | Number of CPU threads |

## Available Models

Download pre-trained checkpoints from the [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2#performance) repository, then convert with `convert_to_gguf.py`.

| Model | Checkpoint | Size |
|-------|-----------|------|
| ViT-Small | `depth_anything_v2_vits.pth` | ~88 MB |
| ViT-Base | `depth_anything_v2_vitb.pth` | ~375 MB |
| ViT-Large | `depth_anything_v2_vitl.pth` | ~1.1 GB |
| ViT-Giant2 | `depth_anything_v2_vitg2.pth` | ~3.5 GB |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DA_DEVICE=cpu` | Force CPU backend |
| `CUDA_VISIBLE_DEVICES` | Select GPU device (when CUDA enabled) |

## Requirements

- CMake 3.12+
- C++17 compiler
- CUDA Toolkit 12.0+ (for GPU acceleration)
- Python 3.8+ (for model conversion only)

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

This project is an independent C++ implementation of the Depth-Anything-V2 inference pipeline. The model architecture and pre-trained weights are derived from [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) (Apache 2.0).

## References

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — Original Python implementation
- [Depth Anything V2 Paper](https://arxiv.org/abs/2406.09414) — Yang et al., 2024
- [ggml](https://github.com/ggml-org/ggml) — Tensor library for ML inference
