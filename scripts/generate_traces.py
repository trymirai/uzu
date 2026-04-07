#!/usr/bin/env python3
"""Generate traces.safetensors for Uzu TraceValidator from HuggingFace Gemma 4.

Captures intermediate activations matching the Uzu trace format:
- Per-layer inputs, norm outputs, attention, MLP, outputs
- Model-level output_norm, logits
- KV cache states

Usage:
  uv run --with torch --with transformers --with safetensors --with accelerate \
    scripts/generate_traces.py
"""

import argparse
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_traces(model_name: str, output_path: str, prompt: str):
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model.eval()

    # Gemma 4 is multimodal — text model is at model.model.language_model
    lm = model.model.language_model

    # Tokenize
    chat = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt")
    input_ids = inputs["input_ids"]
    suffix_length = input_ids.shape[1]

    print(f"Prompt: {repr(formatted[:100])}...")
    print(f"Token IDs: {input_ids[0].tolist()}")
    print(f"Suffix length: {suffix_length}")

    traces = {}
    hooks = []

    # Token info
    traces["activation_trace.token_ids"] = input_ids[0].to(torch.int32)
    traces["activation_trace.token_positions"] = torch.arange(suffix_length, dtype=torch.int32)

    num_layers = len(lm.layers)
    print(f"Model has {num_layers} layers")

    # Storage
    captured = {}

    def hook_output(name):
        """Hook that captures the output tensor."""
        def fn(module, args, output):
            if isinstance(output, tuple):
                captured[name] = output[0].detach().clone().float()
            else:
                captured[name] = output.detach().clone().float()
        return fn

    def hook_output_kw(name):
        """Hook for modules that use kwargs."""
        def fn(module, args, kwargs, output):
            if isinstance(output, tuple):
                captured[name] = output[0].detach().clone().float()
            else:
                captured[name] = output.detach().clone().float()
        return fn

    def hook_input(name):
        """Hook that captures input hidden_states."""
        def fn(module, args, kwargs):
            hs = args[0] if args else kwargs.get('hidden_states')
            if hs is not None:
                captured[name] = hs.detach().clone().float()
        return fn

    # Register hooks per layer
    for i, layer in enumerate(lm.layers):
        # Layer input
        hooks.append(layer.register_forward_pre_hook(hook_input(f"layer.{i}.input"), with_kwargs=True))
        # Layer output
        hooks.append(layer.register_forward_hook(hook_output_kw(f"layer.{i}.output"), with_kwargs=True))
        # Pre-attention norm
        hooks.append(layer.input_layernorm.register_forward_hook(hook_output(f"layer.{i}.pre_mixer_norm")))
        # Attention output
        hooks.append(layer.self_attn.register_forward_hook(hook_output_kw(f"layer.{i}.mixer"), with_kwargs=True))
        # Post-attention norm
        if hasattr(layer, 'post_attention_layernorm'):
            hooks.append(layer.post_attention_layernorm.register_forward_hook(hook_output(f"layer.{i}.post_mixer_norm")))
        # Pre-MLP norm
        hooks.append(layer.pre_feedforward_layernorm.register_forward_hook(hook_output(f"layer.{i}.pre_mlp_norm")))
        # MLP output
        hooks.append(layer.mlp.register_forward_hook(hook_output_kw(f"layer.{i}.mlp"), with_kwargs=True))
        # Post-MLP norm
        if hasattr(layer, 'post_feedforward_layernorm'):
            hooks.append(layer.post_feedforward_layernorm.register_forward_hook(hook_output(f"layer.{i}.post_mlp_norm")))

        # Attention intermediates (layer 0 only, to keep file size manageable)
        if i == 0:
            attn = layer.self_attn
            hooks.append(attn.q_proj.register_forward_hook(hook_output(f"layer.{i}.q_proj")))
            hooks.append(attn.k_proj.register_forward_hook(hook_output(f"layer.{i}.k_proj")))
            hooks.append(attn.v_proj.register_forward_hook(hook_output(f"layer.{i}.v_proj")))
            hooks.append(attn.q_norm.register_forward_hook(hook_output(f"layer.{i}.q_norm")))
            hooks.append(attn.k_norm.register_forward_hook(hook_output(f"layer.{i}.k_norm")))
            hooks.append(attn.v_norm.register_forward_hook(hook_output(f"layer.{i}.v_norm")))
            hooks.append(attn.o_proj.register_forward_hook(hook_output(f"layer.{i}.o_proj")))

            # PLE intermediates
            if hasattr(layer, 'per_layer_input_gate'):
                hooks.append(layer.per_layer_input_gate.register_forward_hook(
                    hook_output(f"layer.{i}.ple_gate")))
                hooks.append(layer.per_layer_projection.register_forward_hook(
                    hook_output(f"layer.{i}.ple_projection")))
                hooks.append(layer.post_per_layer_input_norm.register_forward_hook(
                    hook_output(f"layer.{i}.post_ple_norm")))

    # Output norm
    hooks.append(lm.norm.register_forward_hook(hook_output("output_norm")))

    # Run forward pass
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    logits = outputs.logits.float()
    print(f"Logits shape: {logits.shape}")

    predicted = logits[0].argmax(dim=-1)
    print(f"Predicted next token: '{tokenizer.decode(predicted[-1])}' (id={predicted[-1].item()})")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Build trace tensors (remove batch dim: [1, seq, dim] -> [seq, dim])
    if "output_norm" in captured:
        traces["activation_trace.output_norm"] = captured["output_norm"][0]
    traces["activation_trace.logits"] = logits[0]

    for i in range(num_layers):
        prefix = f"activation_trace.layer_results.{i}"
        ap = f"{prefix}.activation_trace"

        if f"layer.{i}.input" in captured:
            traces[f"{ap}.inputs"] = captured[f"layer.{i}.input"][0]
        if f"layer.{i}.pre_mixer_norm" in captured:
            traces[f"{ap}.pre_mixer_norm"] = captured[f"layer.{i}.pre_mixer_norm"][0]
        if f"layer.{i}.mixer" in captured:
            traces[f"{ap}.mixer"] = captured[f"layer.{i}.mixer"][0]
        if f"layer.{i}.post_mixer_norm" in captured:
            traces[f"{ap}.post_mixer_norm"] = captured[f"layer.{i}.post_mixer_norm"][0]
        if f"layer.{i}.pre_mlp_norm" in captured:
            traces[f"{ap}.pre_mlp_norm"] = captured[f"layer.{i}.pre_mlp_norm"][0]
        if f"layer.{i}.mlp" in captured:
            traces[f"{ap}.mlp"] = captured[f"layer.{i}.mlp"][0]
        if f"layer.{i}.post_mlp_norm" in captured:
            traces[f"{ap}.post_mlp_norm"] = captured[f"layer.{i}.post_mlp_norm"][0]
        if f"layer.{i}.output" in captured:
            traces[f"{prefix}.outputs"] = captured[f"layer.{i}.output"][0]

        for name in ["q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "v_norm", "o_proj",
                     "ple_gate", "ple_projection", "post_ple_norm"]:
            key = f"layer.{i}.{name}"
            if key in captured:
                traces[f"{ap}.{name}"] = captured[key][0]

    # KV cache (DynamicCache API)
    past_kv = outputs.past_key_values
    if past_kv is not None and hasattr(past_kv, 'key_cache'):
        for i in range(min(num_layers, len(past_kv.key_cache))):
            traces[f"updated_kv_cache.{i}.keys"] = past_kv.key_cache[i].float().squeeze(0)
            traces[f"updated_kv_cache.{i}.values"] = past_kv.value_cache[i].float().squeeze(0)

    # Activation tensors should be BF16 to match model precision.
    # Token IDs/positions stay as int32.
    final = {}
    for k, v in traces.items():
        if v.is_floating_point():
            final[k] = v.contiguous().to(torch.bfloat16)
        else:
            final[k] = v.contiguous()

    print(f"\nSaving {len(final)} tensors to {output_path}")
    for k in sorted(final.keys())[:20]:
        print(f"  {k}: {final[k].shape} {final[k].dtype}")
    if len(final) > 20:
        print(f"  ... and {len(final) - 20} more")

    save_file(final, output_path)
    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-e2b-it")
    parser.add_argument("--output", default="models/0.1.8/gemma-4-e2b-4bit/traces.safetensors")
    parser.add_argument("--prompt", default="What is the capital of France?")
    args = parser.parse_args()
    generate_traces(args.model, args.output, args.prompt)


if __name__ == "__main__":
    main()
