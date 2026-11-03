# ANE prefill on M5 — feasibility runbook (for an agent on an M5 Mac)

Goal: decide QUICKLY (≈1–2 h) whether offloading prefill FFNs to the Apple Neural Engine (ANE)
helps on M5, where uzu's GPU already uses the in-GPU matrix units (NAX/"MXU", `supports_mxu`)
with int8 activations × 4-bit weights. Measure first; write code only if the numbers say so.

Repo: uzu, branch `ane-prefill`. Model: `trymirai/Qwen3.5-9B-M` (W4 asym g32, RHT; download with
`hf download trymirai/Qwen3.5-9B-M config.json model.safetensors tokenizer.json --local-dir ~/models/Qwen3.5-9B-M`).
All builds `--release`. Don't push. Report numbers in tables. Mark anything inferred as inferred.

Reference (base M4, 10-core GPU, no MXU): GPU-only prefill 239/229/219 tok/s (1k/2k/4k); with ANE
444/530/555; ANE W8A8 FFN 22.5 ms per 1024 rows per layer; GPU W4 FFN 96.8 ms per 1024 rows per
layer (R = GPU/ANE ≈ 4.3); GPU non-FFN ("mixer") ≈ 33% of GPU-only prefill time.

## Step 0 — environment (5 min)
- `sysctl -n machdep.cpu.brand_string`, `system_profiler SPHardwareDataType` (GPU cores, RAM),
  `sw_vers`, `xcodebuild -version` (if Xcode license/Metal toolchain errors:
  `sudo xcodebuild -license accept` in a real terminal, `xcodebuild -downloadComponent MetalToolchain`).
- Confirm uzu reports MXU: `supportsMXU` is in
  `crates/uzu-engine/src/backends/metal/metal_extensions/device_extensions.rs`; print
  `context.supports_mxu` in a scratch test if unsure.

## Step 1 — is the ANE path even active? (5 min)
On MXU devices `LinearMatmul::prepare_activation` re-tiles W4 codes (`GemmBLayout::MxuTiled`) and
flips them to signed (`matmul_metal_kernel.rs::optimize_weights_layout`). `AneFfn::new` /
`stageable()` in `backends/metal/ane/ffn.rs` then REJECTS the weights, so the engine silently stays
GPU-only. Verify: run the breakdown (Step 2) and check whether any `mlp (ane)` spans appear.
Expected on M5: none. That's fine for Step 2–3; Step 4 explains how to enable it.

## Step 2 — GPU-only baseline + breakdown (15 min)
```
UZU_ANE=0 UZU_PREFILL_MODEL=$HOME/models/Qwen3.5-9B-M UZU_PREFILL_LENGTHS=1024,2048,4096 \
  cargo test --release -p uzu-engine --lib prefill_breakdown -- --exact --ignored --nocapture
```
Record tok/s and per-class GPU time: `layer/mlp (dense)` (= T_gpu_ffn), `layer/delta net`,
`layer/attention`, rest. Compute per 1024-row chunk per layer: T_gpu_ffn, T_gpu_mixer.

## Step 3 — ANE FFN speed on this M5 (10 min)
```
cargo test --release -p uzu-engine --lib ane_ffn_perf_9b_shape -- --ignored --nocapture
```
(9B shape, rows 256/512/768/1024.) Record ms per function → T_ane_ffn (1024 rows). Also note ANE
identity if printed / via `_ANEDeviceInfo` (M4 was `h16g`, 16 cores). Run it twice; report median.

## Step 4 — decide (math, no code)
- R = T_gpu_ffn / T_ane_ffn (per 1024 rows, same layer).
- Pipelined (phase 2) estimate per layer per 1024-row chunk:
  `T_phase2 ≈ max(T_gpu_mixer + T_stage + T_packunpack, T_ane_ffn × 1.1)` with T_stage ≈ 5 ms
  (M4 value; re-measure if the kernel runs, see Step 5) and T_packunpack ≈ 0.5 ms.
- GPU-only per layer per chunk: `T_gpu = T_gpu_mixer + T_gpu_ffn`.
- Expected speedup S = T_gpu / T_phase2. Also compute the intra-layer CHANNEL-SPLIT alternative
  (Splash style; relevant when R is small): optimal FFN time `1/(1/T_gpu_ffn + 1/T_ane_ffn)`,
  S_split = T_gpu / (T_gpu_mixer + that).
- Decision: if max(S, S_split) < 1.15 → stop, report "not worth it on M5" with the numbers.
  If ≥ 1.15 → Step 5. Note: if T_ane_ffn > T_gpu_mixer, pipelining is ANE-bound and the split
  (or a smaller ANE share) is the right shape; if T_ane_ffn ≪ T_gpu_mixer, pipelining wins.

## Step 5 — only if promising: make the ANE path run on M5
Pick the smallest route and measure end-to-end:
1. Quick experiment (no new kernel): disable the MXU re-tiling for the 9B MLP up/down weights
   (e.g. an env flag checked in `LinearMatmul::prepare_activation` that skips
   `optimize_weights_layout`) so `stageable()` accepts them. CAVEAT: this also makes the GPU MLP
   fall back to the non-MXU path, so only the ANE-accepted prefill batches are representative;
   decode would regress — experiment only.
2. Proper: teach `AneStageRows` to read the MXU-tiled, signed-code layout (see
   `GemmKernel::prepare_gemm_b_layout` for the tiling, `signed_codes` = codes XOR 0x8 per nibble),
   or keep a row-major copy of the MLP weights (memory cost ≈ 2.4 GB for 9B).
Then run Step 2's breakdown without `UZU_ANE=0`, plus correctness:
```
UZU_PREFILL_MODEL=... cargo test --release -p uzu-engine --lib prefill_kl -- --ignored --nocapture --skip pipelined
UZU_PREFILL_MODEL=... cargo test --release -p uzu-engine --lib prefill_kl_pipelined -- --ignored --nocapture
```
(KL vs GPU-only must stay ≤ 1e-3; note on M5 the GPU reference itself is A8W4, so compare both to
each other and report.)

## Report
Table: chip/GPU cores/MXU, GPU-only tok/s 1k/2k/4k, T_gpu_ffn, T_gpu_mixer, T_ane_ffn, R, S, S_split,
decision; if Step 5 ran: measured tok/s with ANE, KL, and what was changed.
