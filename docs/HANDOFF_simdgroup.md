# Handoff: simdgroup attention path (`matrix_simd.h`): cleanup to match the M5 design

**From:** the attention-cleanup session (M5 Max machine). **For:** the agent doing simd sliding-window optimizations on another computer.
**Base:** branch `optimize_sliding_attention`, commit `b171b8ffa` ("finish single-split in threadgroup memory").
**Status:** the patches below are being finished now on the M5 machine. Exact diffs come once they land; until then, design to this contract.

## Design rules for every attention backend (MXU, simdgroup, vector)
1. **No split is the normal case:** the core kernel writes the final output directly. A split (≥ 2) is the special case: the core writes partials, and `AttentionMatrixCombine` merges them.
   - Function constant `is_split` (replacing `has_output`; the default is the output path).
   - The host picks the route in one place: `num_splits > 1`.
2. **Bind only what a kernel writes:**
   - `device bfloat* output OPTIONAL(!is_split)`;
   - partial buffers `OPTIONAL(is_split)`.

   Binding a written buffer to a PSO that does not write it makes Metal serialize the split dispatches. It measured +45..+100 % on M5 global decode.
3. **Normalize at the end only:**
   - The core keeps an **un-normalised** accumulator, the row max and the row denominator.
   - The split partial stores {numerator (bf16), max, denominator}.
   - The combine rescales each split by `exp2(split_max - global_max)`, sums, and divides once.
   - Precision against the f32 reference: max |err| identical, mean |err| −15..+0.5 %.
4. **The row math exists once:** `finish_row(numerator, max, denominator)` in `rotated_output.h` (built on `store_output_row`) does sink term, single divide, inverse Hadamard (un-rotate), rotation sign, sigmoid gate and store.
   - Callers: the MXU output path, the combine, the vector direct path, and (TODO, yours) the simdgroup output path.
5. **One mask:** the position-interval mask. A key is visible iff `key_position - oldest < behind + ahead + 1` (unsigned); invalid slots sit at `INVALID_POSITION`.
   - In the simdgroup and vector paths it is the `KeyMask` struct in `mask.h`, replacing `should_use_key`.
   - Window and ring parts are compile-time via `is_sliding_window` / `is_kv_cache_ring`, so global PSOs compile a plain causal compare. Runtime window/ring work cost global/Qwen +1.3..+1.6 % on the M4 Pro.

## What the simdgroup path needs (your scope)
- **Today** `matrix_simd.h` always writes partials (around line 337), even with one split, so every pre-M5 call pays a partial write plus a combine dispatch.
- **Add the output path for `!is_split`:**
  - gather the row across the simdgroups in threadgroup memory (as the MXU path does), then call `finish_row`;
  - reuse existing threadgroup scratch that is dead after the key loop, rather than adding a new array (32 KB limit);
  - the `is_split = true` PSOs should stay byte-identical to today's simdgroup PSOs.
- **Expected:** a speedup on pre-M5 calls with one split (on M5 the same change gave sliding prefill −4..−11 %). Check it on pre-M5 hosts.
- **Coordinate:**
  - `matrix_simd.h`, `vector.metal` and `mask.h` get small edits from the mask patch and the un-normalised accumulator patch (M5 machine).
  - Rebase onto those once they land, and avoid restructuring the same lines in the meantime.

## Measured on the M5 machine (for context)
- **Row-shaped combine** (one simdgroup per row, no threadgroup memory): rejected; batch-1 decode +2..+7 %.
- **No in-core ending** (always core + combine): +7..+13 % sliding prefill. That is why the output path exists.
- **Pre-M5 mask unification:** sliding −5..−12 % on M2 / M4 Pro.

## Update (2026-11-25, after the cleanup landed on this branch)
- **Landed (commit "Attention cleanup: …"):**
  - `is_split` replaces `has_output`: `output` is OPTIONAL(!is_split), partials are OPTIONAL(is_split);
  - PSOs are indexed by mask kind × is_split;
  - the core stores un-normalised partials;
  - `finish_row` (`rotated_output.h`) has three callers: the MXU output path, the combine and the vector direct path;
  - a threadgroup-memory limit check: the no-split output kernel is built only if it fits the device limit (32 KB on M5 Max), otherwise the call takes the split route.
- **For you:**
  - the simdgroup output path (`!is_split`) should call `finish_row` and must respect the same limit check;
  - expected payoff: pre-M5 sliding prefill (it still always pays partials + combine).
- **Pre-M5 findings (M4 Pro, vs b171b8ffa):**
  - The un-normalised accumulator costs global 4k batch 2/8 +2.3/+3.3 % and trie batch 8 +2.2 %; sliding is neutral. Unresolved; the suspect is the simd core or the combine.
  - The single-mask port (`docs/attention_simd_vector_mask.diff`, against b171b8ffa; needs a rebase onto this branch's `vector.metal`):
    - sliding −5.5..−7.8 %;
    - global batch 2/8 at 4k/32k/128k +1.2..+1.8 %;
    - the cause is in `matrix_simd.h`, on the 1- and 4-simdgroup tiles; not found.
- **Inkling decode batch 1 on pre-M5:** runs the vector split kernel (grouped rows ≤ 4). Batch 2 and up run the simd tiles.
