import sys
import json
import os
import struct
import numpy as np
import torch
from tokenizers import Tokenizer
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_vision_forward import (
    preprocess_image,
    build_placeholder_tokens,
    get_rope_index,
    WeightLoader,
    MiniCPMV47VisionTower,
    IMAGE_PAD,
    MODEL_DIR,
)

def prepare_vlm_input(image_path: str, prompt_text: str, output_prefix: str = "/tmp/minicpm_vlm"):
    # 1. Preprocess Image
    patches, grids, slice_grid = preprocess_image(image_path)
    print(f"Preprocessed: {patches.shape}, grids: {len(grids)}, slice_grid: {slice_grid}")

    # 2. Extract Visual Features using Vision Tower
    loader = WeightLoader(MODEL_DIR)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tower = MiniCPMV47VisionTower(loader).to(torch.bfloat16).to(device)
    patches_t = torch.from_numpy(patches).to(torch.bfloat16).to(device)
    with torch.no_grad():
        features, out_grids = tower(patches_t, grids)
    if device == "mps":
        torch.mps.synchronize()

    print(f"Visual features extracted: {features.shape}, dtype: {features.dtype}")
    # Save visual features as raw bytes (BF16)
    features_bytes = features.view(torch.int16).cpu().numpy().tobytes()
    features_bin_path = f"{output_prefix}_features.bin"
    with open(features_bin_path, "wb") as f:
        f.write(features_bytes)

    # 3. Build Prompt Tokens
    tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
    placeholders = build_placeholder_tokens(grids, slice_grid, 0)

    t_prefix = tok.encode("<|im_start|>user\n", add_special_tokens=False).ids
    t_sep = [198] # newline between image and text prompt per Jinja template
    t_query = tok.encode(prompt_text, add_special_tokens=False).ids
    t_suffix = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids

    prompt_tokens = t_prefix + placeholders + t_sep + t_query + t_suffix
    print(f"Total prompt tokens: {len(prompt_tokens)}, image pad count: {prompt_tokens.count(IMAGE_PAD)}")

    visual_token_indices = [idx for idx, tid in enumerate(prompt_tokens) if tid == IMAGE_PAD]
    assert len(visual_token_indices) == features.shape[0], f"Mismatch: {len(visual_token_indices)} pads vs {features.shape[0]} feature rows"

    # 4. Compute Canvas M-RoPE
    pos_3d, delta = get_rope_index(prompt_tokens, grids)
    print(f"M-RoPE coordinates computed: {pos_3d.shape}, delta: {delta}")

    manifest = {
        "prompt_tokens": prompt_tokens,
        "visual_token_indices": visual_token_indices,
        "features_path": features_bin_path,
        "num_visual_tokens": features.shape[0],
        "feature_dim": features.shape[1],
        "mrope_positions": pos_3d.tolist(),
        "delta": int(delta),
    }

    manifest_path = f"{output_prefix}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    print(f"Prepared VLM input successfully -> {manifest_path}")
    return manifest_path

if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "flux-tools.bfl.ai_precise-editing.png"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in detail."
    prepare_vlm_input(img, prompt)
