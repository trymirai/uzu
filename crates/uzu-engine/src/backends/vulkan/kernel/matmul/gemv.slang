import definitions;
import common.activation;
import common.matmul;

// The CPU MatmulKernel's product of one A row and transposed B rows, D = epilogue(A Bᵀ), for every BF16/F32 triple, B
// full precision (Gemv) or U4/U8 quantized (QuantizedGemv), read through common.matmul's weight reader. A workgroup
// computes GEMV_OUTPUTS outputs of one A row, GEMV_LANES invocations each striding K; the partial sums meet in shared
// memory in invocation order, assuming nothing about subgroups. Outputs common.matmul classes careful are summed again
// from exact products, those below 2^-101 summed scaled up by 2^24, where no partial flushes either, and added back with
// one rounding; replayed outputs take the CPU's order.

static const uint GEMV_THREADS = 256;
// Output rows per workgroup, as the binding's group count n.div_ceil(8) assumes.
static const uint GEMV_OUTPUTS = 8;
static const uint GEMV_LANES = GEMV_THREADS / GEMV_OUTPUTS;

groupshared float gemv_main[GEMV_THREADS];
groupshared float gemv_tiny[GEMV_THREADS];
// Per invocation: the matmul_fields of its operands merged.
groupshared uint gemv_fields[GEMV_THREADS];
groupshared uint gemv_mode[GEMV_OUTPUTS];

void gemv_body<A : IModelFloat, B : IModelFloat, D : IModelFloat, W : IMatmulWeights>(
  W weights,
  Ptr<A, Access.Read> a,
  Ptr<D, Access.ReadWrite> d,
  Ptr<B, Access.Read> bias,
  Ptr<uint, Access.Read> gather,
  uint k,
  uint n,
  uint m,
  float ab_scale,
  float soft_cap,
  uint tile,
  uint row,
  uint tid,
  bool has_scale,
  bool accumulate,
  bool has_bias,
  bool has_soft_cap,
  bool gathered,
) {
  let output = tid / GEMV_LANES;
  let lane = tid % GEMV_LANES;
  let column = tile * GEMV_OUTPUTS + output;
  let active = column < n;
  let a_base = row * k;
  uint b_row = 0;
  var partial = 0.0;
  var fields = 0x0000ffffu;
  if (active) {
    b_row = gathered ? gather[row * n + column] : column;
    for (uint j = lane; j < k; j += GEMV_LANES) {
      let x = a[a_base + j].toFloat();
      let w = weights.weight(b_row, j, b_row * k);
      fields = matmul_merge_fields(fields, matmul_fields(x, w));
      partial = ordered_sum(partial, x * w);
    }
  }
  gemv_main[tid] = partial;
  gemv_fields[tid] = fields;
  GroupMemoryBarrierWithGroupSync();

  var dot = 0.0;
  var prior = 0.0;
  var bias_value = 0.0;
  if (lane == 0) {
    var mode = MATMUL_FAST;
    if (active) {
      for (uint other = tid; other < tid + GEMV_LANES; other++) {
        dot = ordered_sum(dot, gemv_main[other]);
        fields = matmul_merge_fields(fields, gemv_fields[other]);
      }
      if (accumulate) {
        prior = d[row * n + column].toFloat();
      }
      if (has_bias) {
        bias_value = bias[column].toFloat();
      }
      mode = matmul_mode(fields, k, has_scale, ab_scale, accumulate, prior, has_bias, bias_value);
    }
    gemv_mode[output] = mode;
  }
  GroupMemoryBarrierWithGroupSync();

  let mode = gemv_mode[output];
  var careful = false;
  for (uint other = 0; other < GEMV_OUTPUTS; other++) {
    careful = careful || gemv_mode[other] == MATMUL_CAREFUL;
  }
  if (careful) {
    var main = 0.0;
    var tiny = 0.0;
    if (mode == MATMUL_CAREFUL) {
      for (uint j = lane; j < k; j += GEMV_LANES) {
        let product = exact_product(a[a_base + j].toFloat(), weights.weight(b_row, j, b_row * k));
        if ((asuint(product) & 0x7fffffffu) >= 0x0d000000u) {
          main = ordered_sum(main, product);
        } else {
          tiny = ordered_sum(tiny, exact_scale_up_24(product));
        }
      }
    }
    gemv_main[tid] = main;
    gemv_tiny[tid] = tiny;
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0 && mode == MATMUL_CAREFUL) {
      main = 0.0;
      tiny = 0.0;
      for (uint other = tid; other < tid + GEMV_LANES; other++) {
        main = ordered_sum(main, gemv_main[other]);
        tiny = ordered_sum(tiny, gemv_tiny[other]);
      }
      dot = exact_sum(main, exact_scale_down(tiny, 24), false);
    }
  }

  if (lane == 0 && active) {
    if (mode == MATMUL_REPLAY) {
      dot = matmul_replay(a, a_base, weights, b_row, k);
    }
    let value = matmul_epilogue(dot, has_scale, ab_scale, accumulate, prior, has_bias, bias_value, has_soft_cap, soft_cap);
    d[row * n + column] = D.fromFloat(value);
  }
}

[[Kernel]] func Gemv<A: IModelFloat, B: IModelFloat, D: IModelFloat>(
  Ptr<B, Access.Read> b,
  Ptr<A, Access.Read> a,
  Ptr<D, Access.ReadWrite> d,
  [[Optional("has_bias")]] Ptr<B, Access.Read> bias,
  [[Optional("gathered")]] Ptr<uint, Access.Read> gather,
  uint k,
  uint n,
  uint m,
  [[Optional("has_scale")]] float ab_scale,
  [[Optional("has_soft_cap")]] float soft_cap,
  [[Groups("n.div_ceil(8)")]] uint tile,
  [[Groups("m")]] uint row,
  [[Threads("256")]] uint tid,
  [[Specialize]] bool has_scale,
  [[Specialize]] bool accumulate,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool has_soft_cap,
  [[Specialize]] bool gathered,
) {
  DenseWeights<B> weights = { b };
  gemv_body<A, B, D>(
    weights, a, d, bias, gather, k, n, m, ab_scale, soft_cap, tile, row, tid, has_scale, accumulate, has_bias,
    has_soft_cap, gathered);
}
