#include <metal_stdlib>
#include "../definitions.metal"
#include "../rng.metal"

using namespace metal;

#define BLOCK_SIZE          1024
#define BLOCK_SIZE_IN_SIMDS (BLOCK_SIZE / 32)
#define MAX_ROUNDS          32

// grain_vec is computed at runtime: ceil(vocab_size / (BLOCK_SIZE * 4))
// This makes the kernel work correctly for any vocab size without recompilation.

// Shared memory layout:
//   [0 .. 7]             CtrlBlock (8 floats) — rejection loop control state
//   [8 .. 8+BLOCK_SIZE-1] working space (raking reduces, per-thread sums, warp aggregates)
#define CTRL_FLOATS  8
#define SHARED_SIZE  (CTRL_FLOATS + BLOCK_SIZE)

struct CtrlBlock {
    float low_logit;    // current lower logit bound (starts at min_p_thresh)
    float high_logit;   // current upper logit bound (starts at max_logit)
    float q_unnorm;     // unnorm mass of tokens above low_logit
    float pivot0_logit; // sampled token's logit (pivot for acceptance check)
    float pfx_before;   // exclusive prefix sum before owner thread
    float owner;        // owning thread index (as float, cast to int)
    float sampled_id;   // result token index (as float, cast to uint)
    float done;         // 0.0 = continue, 1.0 = accepted
};

// ── Per-type vectorized loader ────────────────────────────────────────────────
// Returns 4 floats. Indices base+k >= limit pad with -INFINITY (masked / OOB).
//
// Metal's template resolution does not prefer non-template overloads over templates,
// so a single template handles all three types (float, half, bfloat).
// The fast path (base + 4 <= limit) emits 4 adjacent scalar loads which the LLVM
// backend coalesces into a vector load for float and half.
template <typename T>
static inline float4 load4(device const T* ptr, uint base, uint limit) {
    if (base + 4 <= limit) {
        return float4(float(ptr[base]), float(ptr[base+1]),
                      float(ptr[base+2]), float(ptr[base+3]));
    }
    float4 v(-INFINITY);
    if (base   < limit) v.x = float(ptr[base]);
    if (base+1 < limit) v.y = float(ptr[base+1]);
    if (base+2 < limit) v.z = float(ptr[base+2]);
    if (base+3 < limit) v.w = float(ptr[base+3]);
    return v;
}

// ── Batch-level uniform RNG ───────────────────────────────────────────────────
// All threads with the same (seed, seq) get identical values.
// ctr[2] = 0x8000 places this in a counter space distinct from per-token RNG.
static inline float batch_uniform(uint64_t seed, uint32_t seq) {
    uint key[2] = {uint(seed), uint(seed >> 32)};
    uint ctr[4]  = {seq, 0u, 0x8000u, 0u};
    uint out[4];
    curand_philox4x32_10(ctr, key, out);
    return fmax(float(out[0]) * (1.0f / 4294967296.0f), 1e-37f);
}

// ── Unified stochastic sampling: temperature + top_k/p/min_p + sampling in one dispatch ──
//
// NOTE: No Gumbel noise, no argmax.
//   Gumbel-max (add Gumbel noise to logits → argmax) is mathematically equivalent to
//   inverse-transform sampling from the softmax distribution (draw u ~ U(0,1), find
//   token at CDF position u). This kernel uses the latter: one uniform draw per round,
//   located via a cooperative prefix-sum walk — no per-token noise, no full-vocab argmax.
//
// Algorithm (FlashInfer dual-pivot rejection sampling in logit space):
//
//   Phase 0+1 : stream vocab → cooperative max + sum_exp
//     · min_p = 0 : FUSED single pass using per-thread online softmax
//                   (Milakov & Gimelshein 2018). Each thread accumulates
//                   (local_max, local_sum). After the max reduce, each thread
//                   rescales its local_sum to the global max and the sum reduce
//                   yields the final sum_exp. Saves one full vocab pass.
//     · min_p > 0 : two passes — Phase 0 finds max, Phase 1 sums exp over the
//                   min_p-filtered subset (min_p_thresh = max_logit + log(min_p)).
//
//   Loop (O(1) expected rounds):
//     A) [sampling] Draw u ~ U(0, q_unnorm). Each thread accumulates unnorm mass for
//        tokens with logit >= low_logit into data[thread_idx]. Thread 0 walks the 1024
//        partial sums to find the owning thread; that thread scans its slice to find
//        the exact token. exp() called only for tokens above low_logit.
//     B) [top_k/top_p check] pivot0 = logit[sampled], pivot1 = (pivot0+high_logit)/2.
//        Stream vocab: sum unnorm mass and count for logit > pivot0 / > pivot1.
//     C) [accept or narrow]
//        · ok(pivot0): sampled token satisfies top_k/top_p — accept, break
//        · ok(pivot1): low = pivot0, high = pivot1, q = agg0_unnorm
//        · else       : low = pivot1,               q = agg1_unnorm
//
// Savings vs probability-space pivots:
//   In rounds 2+, low_logit is well above min_p_thresh. Tokens below low_logit
//   are skipped with a single float compare — no exp() call. For typical top_p
//   settings (0.9–0.95), the majority of the vocab is cut after round 1.
//
// Shared memory  : CtrlBlock (8 floats) + BLOCK_SIZE floats (reduce scratch) ≈ 4 KB
// Register usage : O(1) scalars per thread — no per-token register array
//
// Bitmask must be applied via BitmaskKernel before this dispatch.
// Sentinel values to skip filters:
//   top_k = 0     (skip top_k)
//   top_p = 1.0   (skip top_p)
//   min_p = 0.0   (skip min_p, enables fused Phase 0+1)

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(UnifiedStochastic)(
    device const T*        logits,
    device const uint64_t* batch_seeds,
    device uint32_t*       sampled_tokens,
    threadgroup float      shared[SHARED_SIZE],
    constant uint&         batch_size,
    constant uint&         vocab_size,
    constant float&        temperature,
    constant uint&         top_k,
    constant float&        top_p,
    constant float&        min_p,
    const Simd             simd,
    uint batch_idx         GROUPS(batch_size),
    uint thread_idx        THREADS(BLOCK_SIZE)
) {
    const uint     batch_start = batch_idx * vocab_size;
    const uint64_t seed        = batch_seeds[batch_idx];
    const float    inv_temp    = 1.0f / temperature;
    // Number of float4 iterations per thread to cover the full vocab.
    // ceil(vocab_size / (BLOCK_SIZE * 4)) — works for any vocab size.
    const uint     grain_vec   = (vocab_size + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);

    // CtrlBlock lives at the very start; working scratch follows immediately after.
    threadgroup CtrlBlock* ctrl = (threadgroup CtrlBlock*)shared;
    threadgroup float*     data = shared + CTRL_FLOATS;

    // ── Phase 0+1: temperature scaling + max logit + sum_exp ─────────────────
    // Temperature is applied inline via inv_temp on every load4 call below.
    // (No separate temperature kernel dispatch needed.)
    float max_logit, sum_exp, min_p_thresh;

    if (min_p == 0.0f) {
        // ── Fused path: one vocab pass using per-thread online softmax ─────────
        // Each thread builds (local_max, local_sum) where
        //   local_sum = Σ exp(v - local_max) for v in this thread's slice.
        // After the threadgroup max reduce we have global max_logit.
        // Each thread then contributes local_sum * exp(local_max - max_logit)
        // to the final sum reduce — no second vocab pass needed.
        float local_max = -INFINITY;
        float local_sum = 0.0f;
        for (uint i = 0; i < grain_vec; i++) {
            float4 v4 = load4(logits + batch_start,
                              thread_idx * 4 + i * BLOCK_SIZE * 4,
                              vocab_size) * inv_temp;
            for (int k = 0; k < 4; k++) {
                float v = v4[k];
                if (v > -INFINITY) {
                    float new_max = fmax(local_max, v);
                    local_sum = local_sum * fast::exp(local_max - new_max)
                              + fast::exp(v - new_max);
                    local_max = new_max;
                }
            }
        }
        max_logit = threadgroup_raking_reduce_max<BLOCK_SIZE>(
            local_max, data, (ushort)thread_idx);
        // Rescale each thread's partial sum to the global max before summing.
        // local_max = -INFINITY means no valid tokens in this slice → contributes 0.
        float adjusted = (local_max > -INFINITY)
            ? local_sum * fast::exp(local_max - max_logit)
            : 0.0f;
        sum_exp = threadgroup_raking_reduce_sum<BLOCK_SIZE>(
            adjusted, data, (ushort)thread_idx);
        min_p_thresh = -INFINITY;

    } else {
        // ── Two-pass path (min_p > 0) ─────────────────────────────────────────
        // Phase 0: cooperative max
        float local_max = -INFINITY;
        for (uint i = 0; i < grain_vec; i++) {
            float4 v4 = load4(logits + batch_start,
                              thread_idx * 4 + i * BLOCK_SIZE * 4,
                              vocab_size) * inv_temp;
            if (v4.x > -INFINITY) local_max = fmax(local_max, v4.x);
            if (v4.y > -INFINITY) local_max = fmax(local_max, v4.y);
            if (v4.z > -INFINITY) local_max = fmax(local_max, v4.z);
            if (v4.w > -INFINITY) local_max = fmax(local_max, v4.w);
        }
        max_logit    = threadgroup_raking_reduce_max<BLOCK_SIZE>(
            local_max, data, (ushort)thread_idx);
        min_p_thresh = max_logit + fast::log(min_p);

        // Phase 1: sum_exp over min_p-filtered tokens
        float local_sum = 0.0f;
        for (uint i = 0; i < grain_vec; i++) {
            float4 v4 = load4(logits + batch_start,
                              thread_idx * 4 + i * BLOCK_SIZE * 4,
                              vocab_size) * inv_temp;
            if (v4.x > -INFINITY && v4.x >= min_p_thresh) local_sum += fast::exp(v4.x - max_logit);
            if (v4.y > -INFINITY && v4.y >= min_p_thresh) local_sum += fast::exp(v4.y - max_logit);
            if (v4.z > -INFINITY && v4.z >= min_p_thresh) local_sum += fast::exp(v4.z - max_logit);
            if (v4.w > -INFINITY && v4.w >= min_p_thresh) local_sum += fast::exp(v4.w - max_logit);
        }
        sum_exp = threadgroup_raking_reduce_sum<BLOCK_SIZE>(
            local_sum, data, (ushort)thread_idx);
    }

    if (sum_exp <= 0.0f) {
        if (thread_idx == 0) sampled_tokens[batch_idx] = 0;
        return;
    }

    // Precompute top_p threshold in unnorm space to avoid per-round division.
    const float top_p_mass = top_p * sum_exp;

    // ── Initialise control block ──────────────────────────────────────────────
    if (thread_idx == 0) {
        ctrl->sampled_id  = 0.0f;
        ctrl->low_logit   = min_p_thresh;  // logit lower bound
        ctrl->high_logit  = max_logit;     // logit upper bound
        ctrl->q_unnorm    = sum_exp;       // unnorm mass above low_logit
        ctrl->done        = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── FlashInfer dual-pivot rejection loop (logit space) ───────────────────
    for (int round = 0; round < MAX_ROUNDS; round++) {
        const float low_logit  = ctrl->low_logit;
        const float high_logit = ctrl->high_logit;
        const float q_unnorm   = ctrl->q_unnorm;

        // ── A: Sampling (inverse-transform, replaces Gumbel-max + argmax) ───────
        // Draw u ~ U(0, q_unnorm). Find the token at CDF position u.
        // Mathematically equivalent to: add Gumbel noise to each logit → argmax.
        const float u = batch_uniform(seed, (uint)round) * q_unnorm;

        // Each thread accumulates unnorm mass for logit >= low_logit.
        // KEY OPTIMISATION: exp() is only called for tokens above low_logit.
        float local_filtered = 0.0f;
        for (uint i = 0; i < grain_vec; i++) {
            float4 v4 = load4(logits + batch_start,
                              thread_idx * 4 + i * BLOCK_SIZE * 4,
                              vocab_size) * inv_temp;
            if (v4.x > -INFINITY && v4.x >= low_logit) local_filtered += fast::exp(v4.x - max_logit);
            if (v4.y > -INFINITY && v4.y >= low_logit) local_filtered += fast::exp(v4.y - max_logit);
            if (v4.z > -INFINITY && v4.z >= low_logit) local_filtered += fast::exp(v4.z - max_logit);
            if (v4.w > -INFINITY && v4.w >= low_logit) local_filtered += fast::exp(v4.w - max_logit);
        }
        data[thread_idx] = local_filtered;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Thread 0: scan 1024 partial sums to find the owning thread.
        if (thread_idx == 0) {
            float running = 0.0f;
            int   owner   = 0;
            bool  found   = false;
            for (uint t = 0; t < BLOCK_SIZE; t++) {
                float s = data[t];
                if (s > 0.0f) {
                    if (running + s > u) { owner = (int)t; found = true; break; }
                    running += s;
                    owner    = (int)t;
                }
            }
            // Fallback: u slightly exceeded q_unnorm due to FP drift — use last
            // non-zero thread. running currently includes that thread, make exclusive.
            if (!found) running -= data[owner];
            ctrl->owner     = float(owner);
            ctrl->pfx_before = running;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Owner thread: scan its GRAIN elements to find the exact token.
        if ((int)thread_idx == int(ctrl->owner)) {
            const float remaining = fmax(u - ctrl->pfx_before, 0.0f);
            float cum      = 0.0f;
            int   found_id = 0;
            float found_v  = low_logit;
            bool  got_any  = false;
            bool  stop     = false;
            for (uint i = 0; i < grain_vec && !stop; i++) {
                uint base = thread_idx * 4 + i * BLOCK_SIZE * 4;
                for (int k = 0; k < 4 && !stop; k++) {
                    uint vi = base + k;
                    if (vi < vocab_size) {
                        float v = float(logits[batch_start + vi]) * inv_temp;
                        if (v > -INFINITY && v >= low_logit) {
                            float unnorm = fast::exp(v - max_logit);
                            if (!got_any) { found_id = (int)vi; found_v = v; got_any = true; }
                            cum += unnorm;
                            if (cum > remaining) { found_id = (int)vi; found_v = v; stop = true; }
                        }
                    }
                }
            }
            ctrl->sampled_id   = float(found_id);
            ctrl->pivot0_logit = found_v;   // logit value, not probability
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Early exit when no top_k/top_p filtering needed.
        if (top_k == 0 && top_p >= 1.0f) break;

        // ── B: Aggregate unnorm mass and count above each logit pivot ─────────
        // Tokens below low_logit are skipped entirely (no exp needed).
        const float pivot0 = ctrl->pivot0_logit;
        const float pivot1 = (pivot0 + high_logit) * 0.5f;

        float la0 = 0.0f, lc0 = 0.0f, la1 = 0.0f, lc1 = 0.0f;
        for (uint i = 0; i < grain_vec; i++) {
            float4 v4 = load4(logits + batch_start,
                              thread_idx * 4 + i * BLOCK_SIZE * 4,
                              vocab_size) * inv_temp;
            for (int k = 0; k < 4; k++) {
                float v = v4[k];
                if (v > -INFINITY && v >= low_logit) {
                    float unnorm = fast::exp(v - max_logit);
                    if (v > pivot0) { la0 += unnorm; lc0 += 1.0f; }
                    if (v > pivot1) { la1 += unnorm; lc1 += 1.0f; }
                }
            }
        }

        // Combined 4-value reduce (2 barriers).
        if (simd.lane_idx == 0) {
            uint g = simd.group_idx;
            data[g * 4 + 0] = simd_sum(la0);
            data[g * 4 + 1] = simd_sum(lc0);
            data[g * 4 + 2] = simd_sum(la1);
            data[g * 4 + 3] = simd_sum(lc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_idx == 0) {
            float agg0 = 0.0f, cnt0 = 0.0f, agg1 = 0.0f, cnt1 = 0.0f;
            for (uint g = 0; g < BLOCK_SIZE_IN_SIMDS; g++) {
                agg0 += data[g * 4 + 0];
                cnt0 += data[g * 4 + 1];
                agg1 += data[g * 4 + 2];
                cnt1 += data[g * 4 + 3];
            }

            // ── C: Accept or narrow ───────────────────────────────────────────
            // top_p check in unnorm space: agg_unnorm < top_p * sum_exp
            const bool ok_k0 = (top_k == 0) || (cnt0 < float(top_k));
            const bool ok_p0 = (top_p >= 1.0f) || (agg0 < top_p_mass);
            const bool ok_k1 = (top_k == 0) || (cnt1 < float(top_k));
            const bool ok_p1 = (top_p >= 1.0f) || (agg1 < top_p_mass);

            if (ok_k0 && ok_p0) {
                ctrl->done = 1.0f;
            } else if (ok_k1 && ok_p1) {
                ctrl->low_logit  = pivot0;
                ctrl->high_logit = pivot1;
                ctrl->q_unnorm   = agg0;
            } else {
                ctrl->low_logit = pivot1;
                ctrl->q_unnorm  = agg1;
                // high_logit unchanged
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (ctrl->done != 0.0f) break;
    }

    if (thread_idx == 0) {
        sampled_tokens[batch_idx] = uint(ctrl->sampled_id);
    }
}
