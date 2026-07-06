#!/usr/bin/env python3
# Copyright 2026 depth-anything-v2.cpp contributors
# Licensed under the Apache License, Version 2.0
"""Convert Depth-Anything-V2 .pth checkpoint to GGUF format using gguf-py."""
import sys
import os
import argparse
import numpy as np
import torch
from gguf import GGUFWriter, GGUFValueType

def convert_checkpoint(checkpoint_path, output_path, quantize="f32"):
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Determine model config
    embed_dim = None
    if "pretrained.patch_embed.proj.weight" in state_dict:
        embed_dim = state_dict["pretrained.patch_embed.proj.weight"].shape[0]

    max_block = 0
    for key in state_dict:
        if key.startswith("pretrained.blocks."):
            idx = int(key.split(".")[2])
            max_block = max(max_block, idx)
    depth = max_block + 1 if max_block > 0 else 0

    num_heads = 0
    if embed_dim and "pretrained.blocks.0.attn.qkv.weight" in state_dict:
        num_heads = embed_dim // 64

    arch_map = {384: "vits", 768: "vitb", 1024: "vitl", 1536: "vitg"}
    arch = arch_map.get(embed_dim, "unknown")

    head_features = 0
    if "depth_head.scratch.layer1_rn.weight" in state_dict:
        head_features = state_dict["depth_head.scratch.layer1_rn.weight"].shape[0]

    out_channels = [int(x) for x in []]
    for i in range(4):
        key = f"depth_head.scratch.layer{i+1}_rn.weight"
        if key in state_dict:
            out_channels.append(int(state_dict[key].shape[0]))

    intermediate_map = {
        "vits": [2, 5, 8, 11], "vitb": [2, 5, 8, 11],
        "vitl": [4, 11, 17, 23], "vitg": [9, 19, 29, 39],
    }
    intermediate_layers = intermediate_map.get(arch, [2, 5, 8, 11])
    ffn_layer = "swiglufused" if arch == "vitg" else "mlp"

    print(f"Detected: arch={arch}, embed_dim={embed_dim}, depth={depth}, "
          f"num_heads={num_heads}, head_features={head_features}")

    gguf = GGUFWriter(output_path, "Depth Anything V2")

    gguf.add_key_value("general.architecture", "depth_anything_v2", GGUFValueType.STRING)
    gguf.add_key_value("general.name", "Depth Anything V2", GGUFValueType.STRING)
    gguf.add_key_value("depth_anything.embed_dim", embed_dim, GGUFValueType.UINT32)
    gguf.add_key_value("depth_anything.depth", depth, GGUFValueType.UINT32)
    gguf.add_key_value("depth_anything.arch", arch, GGUFValueType.STRING)
    gguf.add_key_value("depth_anything.num_heads", num_heads, GGUFValueType.UINT32)
    gguf.add_key_value("depth_anything.mlp_ratio", 4, GGUFValueType.UINT32)
    gguf.add_key_value("depth_anything.patch_size", 14, GGUFValueType.UINT32)
    gguf.add_key_value("depth_anything.input_size", 518, GGUFValueType.UINT32)
    gguf.add_key_value("depth_anything.init_values", 1.0, GGUFValueType.FLOAT32)
    gguf.add_key_value("depth_anything.ffn_layer", ffn_layer, GGUFValueType.STRING)
    gguf.add_key_value("depth_anything.interpolate_offset", 0.1, GGUFValueType.FLOAT32)
    gguf.add_key_value("depth_anything.interpolate_antialias", False, GGUFValueType.BOOL)
    gguf.add_key_value("depth_anything.head_features", head_features, GGUFValueType.UINT32)
    gguf.add_key_value("depth_anything.out_channels", out_channels, GGUFValueType.ARRAY, GGUFValueType.UINT32)
    gguf.add_key_value("depth_anything.intermediate_layers", intermediate_layers, GGUFValueType.ARRAY, GGUFValueType.INT32)

    # Add tensors
    # Store tensors as-is (PyTorch order, C-contiguous).
    # GGUF stores shape as data.shape and bytes in C-contiguous (last-dim-fastest).
    # ggml reverses shape on load: [cout,cin,kh,kw] -> ne=[kw,kh,cin,cout].
    # Since ggml expects first-dim-fastest (ne[0] fastest), and GGUF stores
    # last-dim-fastest, the reversal makes them align perfectly.
    # For 2D: PyTorch [out,in] -> ggml ne=[in,out] ✓
    print(f"Adding {len(state_dict)} tensors...")
    for name in sorted(state_dict.keys()):
        if "mask_token" in name:
            continue
        tensor = state_dict[name]
        data = tensor.detach().cpu().numpy().astype(np.float32)
        # Ensure conv2d weights are always 4D [cout, cin, kh, kw]
        if "conv" in name and "weight" in name and data.ndim < 4:
            while data.ndim < 4:
                data = np.expand_dims(data, -1)
        if quantize == "f16":
            data = data.astype(np.float16)
        gguf.add_tensor(name, data)

    gguf.write_header_to_file()
    gguf.write_kv_data_to_file()
    gguf.write_tensors_to_file()
    gguf.close()

    file_size = os.path.getsize(output_path)
    print(f"Done! File size: {file_size / (1024*1024):.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Convert Depth-Anything-V2 checkpoint to GGUF")
    parser.add_argument("checkpoint", help="Path to .pth checkpoint file")
    parser.add_argument("output", help="Output GGUF file path")
    parser.add_argument("--quantize", choices=["f32", "f16"], default="f32")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: not found: {args.checkpoint}")
        sys.exit(1)

    convert_checkpoint(args.checkpoint, args.output, args.quantize)


if __name__ == "__main__":
    main()
