#include <metal_stdlib>
#include "../../activation/activations.h"
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../common/integral_constant.h"
#include "../../common/thread_context.h"
#include "../../generated/trie.h"
#include "../../matmul/common/fragment.h"
#include "../../matmul/common/simdgroup_fragment_ops.h"
#include "../common/tri_inv.h"

using namespace metal;
using namespace uzu::matmul;
using namespace uzu::trie;

#define HEAD_V_DIM 128u
#define HEAD_K_DIM 128u
#define BLOCK_SIZE 16u
#define OWNERS 8u
#define OWNER_COLS (HEAD_V_DIM / OWNERS)
#define OUT_STRIDE (HEAD_V_DIM + 1u)

using GramAccFragment = Fragment<float, 2, 2, SimdgroupFragmentOps>;

// Verifies a whole tree (NB = 1: up to 16 nodes, NB = 2: up to 32) per (batch, value head). Simdgroups 0 .. NB - 1
// build the grams and invert I + A; the other eight own 16 value columns each.
template <typename T, uint NB>
VARIANTS(T, float, bfloat)
VARIANTS(NB, 1, 2)
PUBLIC KERNEL(TreeVerify)(
    const device T* q,
    const device T* k,
    const device T* v,
    const device TrieNode* trie,
    const device float* log_decay,
    const device float* beta,
    const device float* h0,
    const device T* in_proj,
    const device float* norm_weight,
    device T* o,
    constant const float& scale,
    constant const float& norm_epsilon,
    constant const uint& batch_size,
    constant const uint& tree_size,
    constant const uint& num_k_heads,
    constant const uint& num_v_heads,
    constant const uint& head_k_dim,
    constant const uint& head_v_dim,
    constant const uint& conv_dim,
    constant const uint& total_proj_dim,
    threadgroup T out_tile[NB * BLOCK_SIZE * OUT_STRIDE],
    threadgroup float a_tiles[NB * NB * BLOCK_SIZE * BLOCK_SIZE],
    threadgroup float qkd_tile[NB * NB * BLOCK_SIZE * BLOCK_SIZE],
    threadgroup float diag_inverses[NB * BLOCK_SIZE * BLOCK_SIZE],
    threadgroup float row_prefix[NB * BLOCK_SIZE],
    threadgroup float row_beta[NB * BLOCK_SIZE],
    threadgroup float row_decay[NB * BLOCK_SIZE],
    threadgroup uint trie_start[NB * BLOCK_SIZE],
    threadgroup uint trie_end[NB * BLOCK_SIZE],
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint hv_idx GROUPS(num_v_heads),
    const uint thread_idx THREADS((OWNERS + NB) * METAL_SIMD_SIZE)
) {
  using Ops = SimdgroupFragmentOps;
  constexpr uint TOKENS = NB * BLOCK_SIZE;
  constexpr uint SIMDGROUPS = OWNERS + NB;
  constexpr ushort TOKEN_FRAGMENTS = TOKENS / Ops::FRAGMENT_COLS;
  using OwnerFragment = Fragment<float, OWNER_COLS / Ops::FRAGMENT_ROWS, TOKEN_FRAGMENTS, Ops>;
  using BlockRightFragment = OperandFragment<float, TOKEN_FRAGMENTS, TOKEN_FRAGMENTS, Ops, ReadTranspose>;

  const ushort lane = thread_context.simd_lane_id;
  const uint simdgroup_idx = thread_context.simdgroup_index;
  const bool is_owner = simdgroup_idx >= NB;
  // DeltaNet::new rejects head_k_dim != 128; NB = 1 takes the constant so the K guards fold away.
  const uint k_dim = NB == 1 ? HEAD_K_DIM : head_k_dim;
  const int qk_stride = num_k_heads * k_dim;
  const uint qk_base = (batch_idx * tree_size * num_k_heads + hv_idx / (num_v_heads / num_k_heads)) * k_dim;
  const uint scalar_base = batch_idx * tree_size * num_v_heads + hv_idx;
  const uint value_token_stride = num_v_heads * head_v_dim;
  const uint value_base = batch_idx * tree_size * value_token_stride + hv_idx * head_v_dim;
  const uint dv_base = (simdgroup_idx - NB) * OWNER_COLS;
  const short rows_left = short(tree_size);
  const auto load_rows = [&](auto full, thread auto& frag, const device T* base, const short rows) UZU_ALWAYS_INLINE {
    auto source = fragment_source(base, qk_stride);
    if constexpr (decltype(full)::value) {
      frag.load_from(lane, source);
    } else {
      frag.load_from(lane, source.bounded(rows, Ops::FRAGMENT_ROWS));
    }
  };

  OwnerFragment v_frag;
  if (is_owner) {
    v_frag.load_from(
        lane,
        fragment_source(v + value_base + dv_base, 1, int(value_token_stride)).bounded(OWNER_COLS, rows_left)
    );
  }

  {
    const bool lane_valid = lane < tree_size;
    const TrieNode node = lane_valid ? trie[batch_idx * tree_size + lane] : TrieNode{1, 0, 0};
    const float lane_log_decay = lane_valid ? log_decay[scalar_base + lane * num_v_heads] : 0.0f;
    if (simdgroup_idx == 0 && lane < TOKENS) {
      trie_start[lane] = node.trie_start;
      trie_end[lane] = node.trie_end;
    }
    for (uint row = simdgroup_idx; row < TOKENS; row += SIMDGROUPS) {
      const float sum = simd_sum((row >= node.trie_start && row <= node.trie_end) ? lane_log_decay : 0.0f);
      if (lane == 0) {
        const bool row_valid = row < tree_size;
        row_prefix[row] = row_valid ? sum : 0.0f;
        row_decay[row] = row_valid ? exp(sum) : 1.0f;
        row_beta[row] = row_valid ? beta[scalar_base + row * num_v_heads] : 0.0f;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  OwnerFragment kh0;
  OwnerFragment qh0;
  kh0.clear();
  qh0.clear();
  if (!is_owner) {
    // Owners read all of qkd, so its upper blocks are zeroed.
    auto build_gram_blocks = [&](auto builder) UZU_ALWAYS_INLINE {
      constexpr uint B = decltype(builder)::value;
      GramAccFragment diag_a;
      GramAccFragment diag_q;
      GramAccFragment extra;
      const device T* k_rows = k + qk_base + B * BLOCK_SIZE * qk_stride;
      const device T* q_rows = q + qk_base + B * BLOCK_SIZE * qk_stride;
      const short rows = rows_left - short(B * BLOCK_SIZE);
      uzu::dispatch_bool(rows_left >= short(TOKENS), [&](auto full) UZU_ALWAYS_INLINE {
        using GramLeftFragment = OperandFragment<float, 2, 1, Ops>;
        using GramRightFragment = OperandFragment<float, 1, 2, Ops, ReadTranspose>;
        diag_a.clear();
        diag_q.clear();
        extra.clear();
        _Pragma("clang loop unroll_count(2)") for (ushort step = 0; step < HEAD_K_DIM / Ops::FRAGMENT_ROWS; ++step) {
          const uint kb = step * Ops::FRAGMENT_ROWS;
          if (kb < k_dim) {
            GramLeftFragment k_left;
            GramLeftFragment q_left;
            GramRightFragment k_right;
            load_rows(full, k_left, k_rows + kb, rows);
            load_rows(full, q_left, q_rows + kb, rows);
            load_rows(full, k_right, k_rows + kb, rows);
            fragment_mma(diag_a, k_left, k_right);
            fragment_mma(diag_q, q_left, k_right);
            if constexpr (NB == 2 && B == 0) {
              GramLeftFragment k_below;
              load_rows(full, k_below, k_rows + BLOCK_SIZE * qk_stride + kb, rows - BLOCK_SIZE);
              fragment_mma(extra, k_below, k_right);
            } else if constexpr (NB == 2) {
              GramRightFragment k_first;
              load_rows(full, k_first, k + qk_base + kb, rows_left);
              fragment_mma(extra, q_left, k_first);
            }
          }
        }
      });
      auto store = [&](thread GramAccFragment& block, const bool is_qkd, const uint i, const uint j) UZU_ALWAYS_INLINE {
        block.map_coords(lane, [&](short r, short c, float value) UZU_ALWAYS_INLINE {
          const uint row = i * BLOCK_SIZE + r;
          const uint col = j * BLOCK_SIZE + c;
          const bool in_subtree = row >= trie_start[col] && row <= trie_end[col];
          const float decay = exp(row_prefix[row] - row_prefix[col]);
          if (is_qkd) {
            return in_subtree ? scale * decay * value : 0.0f;
          }
          return in_subtree && row != col ? row_beta[row] * decay * value : 0.0f;
        });
        block.store(lane, (is_qkd ? qkd_tile : a_tiles) + i * BLOCK_SIZE * TOKENS + j * BLOCK_SIZE, int(TOKENS));
      };
      store(diag_a, false, B, B);
      store(diag_q, true, B, B);
      if constexpr (NB == 2) {
        store(extra, B == 1, 1, 0);
        if constexpr (B == 0) {
          GramAccFragment zero;
          zero.clear();
          zero.store(lane, qkd_tile + BLOCK_SIZE, int(TOKENS));
        }
      }
    };
    if (simdgroup_idx == 0) {
      build_gram_blocks(metal::integral_constant<uint, 0>{});
    } else if constexpr (NB == 2) {
      build_gram_blocks(metal::integral_constant<uint, 1>{});
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    invert_lower_triangular_block<BLOCK_SIZE, TOKENS>(
        diag_inverses + simdgroup_idx * (BLOCK_SIZE * BLOCK_SIZE),
        a_tiles + simdgroup_idx * BLOCK_SIZE * (TOKENS + 1),
        tree_size - simdgroup_idx * BLOCK_SIZE,
        lane
    );
  } else {
    const device float* h0_rows = h0 + ((batch_idx * num_v_heads + hv_idx) * head_v_dim + dv_base) * k_dim;
    uzu::dispatch_bool(rows_left >= short(TOKENS), [&](auto full) UZU_ALWAYS_INLINE {
      using TokenRightFragment = OperandFragment<float, 1, TOKEN_FRAGMENTS, Ops, ReadTranspose>;
      METAL_PRAGMA_UNROLL
      for (ushort step = 0; step < HEAD_K_DIM / Ops::FRAGMENT_ROWS; ++step) {
        const uint kb = step * Ops::FRAGMENT_ROWS;
        if (kb < k_dim) {
          OperandFragment<float, 2, 1, Ops> h0_frag;
          TokenRightFragment k_right;
          TokenRightFragment q_right;
          h0_frag.load_from(lane, fragment_source(h0_rows + kb, int(k_dim)));
          load_rows(full, k_right, k + qk_base + kb, rows_left);
          load_rows(full, q_right, q + qk_base + kb, rows_left);
          fragment_mma(kh0, h0_frag, k_right);
          fragment_mma(qh0, h0_frag, q_right);
        }
      }
    });
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Builder b solves column slice b in place over A; only builder 0 reads the off-diagonal block it overwrites.
  const threadgroup float* ainv = diag_inverses;
  if constexpr (NB > 1) {
    ainv = a_tiles;
    if (!is_owner) {
      solve_lower_triangular_slice<BLOCK_SIZE, NB, float>(
          [&](uint b)
              UZU_ALWAYS_INLINE { return static_cast<const threadgroup float*>(a_tiles + b * BLOCK_SIZE * TOKENS); },
          [](const threadgroup float* row, uint m) UZU_ALWAYS_INLINE { return row + m * BLOCK_SIZE; },
          TOKENS,
          diag_inverses,
          [&](uint b, thread GramAccFragment& t_block) UZU_ALWAYS_INLINE {
            t_block.store(lane, a_tiles + b * BLOCK_SIZE * TOKENS + simdgroup_idx * BLOCK_SIZE, int(TOKENS));
          },
          simdgroup_idx,
          lane
      );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (is_owner) {
    OwnerFragment::zip_for_each_coord(
        lane,
        [&](ushort, ushort token, thread float& kh0_value, thread float& v_value)
            UZU_ALWAYS_INLINE { kh0_value = row_beta[token] * (v_value - row_decay[token] * kh0_value); },
        kh0,
        v_frag
    );
    BlockRightFragment ainv_right;
    ainv_right.load_from(lane, fragment_source(ainv, int(TOKENS)));
    OwnerFragment u;
    u.clear();
    fragment_mma(u, kh0, ainv_right);

    qh0.map_coords(lane, [&](short, short token, float value) UZU_ALWAYS_INLINE {
      return value * (row_decay[token] * scale);
    });
    BlockRightFragment qkd_right;
    qkd_right.load_from(lane, fragment_source(qkd_tile, int(TOKENS)));
    fragment_mma(qh0, u, qkd_right);
    // Rounded to T before the norm.
    qh0.map_coords(lane, [&](short value_col, short token, float value) UZU_ALWAYS_INLINE {
      out_tile[token * OUT_STRIDE + dv_base + value_col] = static_cast<T>(value);
      return value;
    });
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  using T4 = metal::vec<T, 4>;
  const uint dv = lane * 4;
  for (uint token = simdgroup_idx; token < tree_size; token += SIMDGROUPS) {
    float values[4];
    float square_sum = 0.0f;
    METAL_PRAGMA_UNROLL
    for (uint j = 0; j < 4; ++j) {
      values[j] = float(out_tile[token * OUT_STRIDE + dv + j]);
      square_sum += values[j] * values[j];
    }
    const float inv_rms = rsqrt(simd_sum(square_sum) / float(head_v_dim) + norm_epsilon);
    const device T* z_row =
        in_proj + (batch_idx * tree_size + token) * total_proj_dim + conv_dim + hv_idx * head_v_dim + dv;
    T4 result;
    METAL_PRAGMA_UNROLL
    for (uint j = 0; j < 4; ++j) {
      result[j] = static_cast<T>(values[j] * inv_rms * norm_weight[dv + j] * activate_silu(float(z_row[j])));
    }
    *reinterpret_cast<device T4*>(o + value_base + token * value_token_stride + dv) = result;
  }
}
