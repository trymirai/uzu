import definitions;
import common.activation;
import common.matmul;

// The CPU MatmulKernel's full-precision product of A rows and transposed contiguous B rows, D = epilogue(A Bᵀ), for
// every BF16/F32 triple. A workgroup computes a GEMM_TILE x GEMM_TILE tile of D from shared slices of GEMM_DEPTH K
// elements of A's and B's tile rows; each invocation owns GEMM_SPAN x GEMM_SPAN outputs GEMM_STRIDE apart and folds
// each one's products from +0 in increasing K, the CPU's order, a NoContraction product and sum per step. Where
// common.matmul classes an output fast that fold sees no operand, product or partial the device may flush, so it rounds
// every step as the CPU does; every other output replays the CPU's order. Out-of-range elements load as +0, which adds
// nothing to a fold from +0 and nothing to the exponent fields, so every invocation passes every barrier.

static const uint GEMM_THREADS = 256;
// Rows and columns of D per workgroup, as the binding's group counts m.div_ceil(64) and n.div_ceil(64) assume.
static const uint GEMM_TILE = 64;
static const uint GEMM_DEPTH = 16;
static const uint GEMM_STRIDE = 16;
static const uint GEMM_SPAN = GEMM_TILE / GEMM_STRIDE;

// A's and B's tile rows of one slice, depth-major: [depth * GEMM_TILE + tile row].
groupshared float gemm_a[GEMM_DEPTH * GEMM_TILE];
groupshared float gemm_b[GEMM_DEPTH * GEMM_TILE];
// Per invocation: the matmul_fields of the A and B elements it loaded.
groupshared uint gemm_a_fields[GEMM_THREADS];
groupshared uint gemm_b_fields[GEMM_THREADS];

// sum + x w, the product and the sum each rounded natively, never fused.
float gemm_step(float sum, float x, float w) {
  return spirv_asm {
    %product:$$float = OpFMul $x $w;
    OpDecorate %product NoContraction;
    %sum:$$float = OpFAdd $sum %product;
    OpDecorate %sum NoContraction;
    result:$$float = OpCopyObject %sum;
  };
}

[[Kernel]] func Gemm<A: IModelFloat, B: IModelFloat, D: IModelFloat>(
  Ptr<B, Access.Read> b,
  Ptr<A, Access.Read> a,
  Ptr<D, Access.ReadWrite> d,
  [[Optional("has_bias")]] Ptr<B, Access.Read> bias,
  uint k,
  uint n,
  uint m,
  [[Optional("has_scale")]] float ab_scale,
  [[Optional("has_soft_cap")]] float soft_cap,
  [[Groups("n.div_ceil(64)")]] uint column_tile,
  [[Groups("m.div_ceil(64)")]] uint row_tile,
  [[Threads("256")]] uint tid,
  [[Specialize]] bool has_scale,
  [[Specialize]] bool accumulate,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool has_soft_cap,
) {
  let first_row = row_tile * GEMM_TILE;
  let first_column = column_tile * GEMM_TILE;
  // Each slice, this invocation loads GEMM_SPAN consecutive K elements from `slot` of tile row `loaded` of A and of B.
  let loaded = tid / GEMM_SPAN;
  let slot = tid % GEMM_SPAN * GEMM_SPAN;
  let a_row = first_row + loaded;
  let b_row = first_column + loaded;
  // It owns tile rows owner_row + GEMM_STRIDE i and tile columns owner_column + GEMM_STRIDE j.
  let owner_row = tid / GEMM_STRIDE;
  let owner_column = tid % GEMM_STRIDE;
  // Row groups from here hold no row of A in every invocation alike.
  let live_rows = m - first_row;
  float sums[GEMM_SPAN][GEMM_SPAN];
  [unroll]
  for (uint i = 0; i < GEMM_SPAN; i++) {
    [unroll]
    for (uint j = 0; j < GEMM_SPAN; j++) {
      sums[i][j] = 0.0;
    }
  }
  var a_fields = 0x0000ffffu;
  var b_fields = 0x0000ffffu;
  for (uint base = 0; base < k; base += GEMM_DEPTH) {
    [unroll]
    for (uint q = 0; q < GEMM_SPAN; q++) {
      let j = base + slot + q;
      var x = 0.0;
      var w = 0.0;
      if (a_row < m && j < k) {
        x = a[a_row * k + j].toFloat();
      }
      if (b_row < n && j < k) {
        w = b[b_row * k + j].toFloat();
      }
      a_fields = matmul_merge_fields(a_fields, matmul_fields(x, 0.0));
      b_fields = matmul_merge_fields(b_fields, matmul_fields(0.0, w));
      gemm_a[(slot + q) * GEMM_TILE + loaded] = x;
      gemm_b[(slot + q) * GEMM_TILE + loaded] = w;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint depth = 0; depth < GEMM_DEPTH; depth++) {
      float w[GEMM_SPAN];
      [unroll]
      for (uint j = 0; j < GEMM_SPAN; j++) {
        w[j] = gemm_b[depth * GEMM_TILE + owner_column + GEMM_STRIDE * j];
      }
      [unroll]
      for (uint i = 0; i < GEMM_SPAN; i++) {
        if (GEMM_STRIDE * i < live_rows) {
          let x = gemm_a[depth * GEMM_TILE + owner_row + GEMM_STRIDE * i];
          [unroll]
          for (uint j = 0; j < GEMM_SPAN; j++) {
            sums[i][j] = gemm_step(sums[i][j], x, w[j]);
          }
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  gemm_a_fields[tid] = a_fields;
  gemm_b_fields[tid] = b_fields;
  GroupMemoryBarrierWithGroupSync();

  [unroll]
  for (uint i = 0; i < GEMM_SPAN; i++) {
    [unroll]
    for (uint j = 0; j < GEMM_SPAN; j++) {
      let tile_row = owner_row + GEMM_STRIDE * i;
      let tile_column = owner_column + GEMM_STRIDE * j;
      let row = first_row + tile_row;
      let column = first_column + tile_column;
      if (row < m && column < n) {
        var fields = 0x0000ffffu;
        for (uint part = 0; part < GEMM_SPAN; part++) {
          fields = matmul_merge_fields(fields, gemm_a_fields[tile_row * GEMM_SPAN + part]);
          fields = matmul_merge_fields(fields, gemm_b_fields[tile_column * GEMM_SPAN + part]);
        }
        var prior = 0.0;
        if (accumulate) {
          prior = d[row * n + column].toFloat();
        }
        var bias_value = 0.0;
        if (has_bias) {
          bias_value = bias[column].toFloat();
        }
        var dot = sums[i][j];
        if (matmul_mode(fields, k, has_scale, ab_scale, accumulate, prior, has_bias, bias_value) != MATMUL_FAST) {
          dot = matmul_replay(a, row * k, b, column * k, k);
        }
        let value =
          matmul_epilogue(dot, has_scale, ab_scale, accumulate, prior, has_bias, bias_value, has_soft_cap, soft_cap);
        d[row * n + column] = D.fromFloat(value);
      }
    }
  }
}
