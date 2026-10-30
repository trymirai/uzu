#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"

using namespace metal;

#define LANES 32u
#define QUADS_MAX 16u
#define ROWS_MAX 1024u
#define ABSENT 7u

PUBLIC KERNEL(ConfidenceHeadTreeAttention)(
    const device bfloat* qkv,
    const device uint* relations,
    const device bfloat* relation_bias,
    device bfloat* output,
    const constant uint& rows,
    const uint num_heads SPECIALIZE,
    const uint head_dim SPECIALIZE,
    threadgroup float weights[ROWS_MAX],
    const uint index GROUPS(rows * num_heads),
    const uint lane THREADS(LANES)
) {
  const uint query = index / num_heads;
  const uint head = index % num_heads;
  const uint width = num_heads * head_dim;
  const uint quads = head_dim / 4u;
  const device uint* relation = relations + query * rows;
  device bfloat* out = output + query * width + head * head_dim;
  if (relation[query] == ABSENT) {
    for (uint unit = lane; unit < head_dim; unit += LANES) {
      out[unit] = bfloat(0.0f);
    }
    return;
  }
  const float scale = rsqrt(float(head_dim));
  const device bfloat4* q = reinterpret_cast<const device bfloat4*>(qkv + query * 3 * width + head * head_dim);
  float4 query_values[QUADS_MAX];
  METAL_PRAGMA_UNROLL
  for (uint quad = 0; quad < QUADS_MAX; ++quad) {
    query_values[quad] = quad < quads ? float4(q[quad]) * scale : float4(0.0f);
  }

  float local_max = -INFINITY;
  for (uint key = lane; key < rows; key += LANES) {
    const uint code = relation[key];
    float score = -INFINITY;
    if (code != ABSENT) {
      const device bfloat4* k = reinterpret_cast<const device bfloat4*>(qkv + key * 3 * width + width + head * head_dim);
      score = code == 0 ? 0.0f : float(relation_bias[head * 6 + code - 1]);
      METAL_PRAGMA_UNROLL
      for (uint quad = 0; quad < QUADS_MAX; ++quad) {
        score += quad < quads ? dot(query_values[quad], float4(k[quad])) : 0.0f;
      }
    }
    weights[key] = score;
    local_max = max(local_max, score);
  }
  const float top = simd_max(local_max);
  float local_total = 0.0f;
  for (uint key = lane; key < rows; key += LANES) {
    const float weight = exp(weights[key] - top);
    weights[key] = weight;
    local_total += weight;
  }
  const float total = simd_sum(local_total);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint group = lane % quads;
  const uint split = lane / quads;
  const uint splits = LANES / quads;
  float4 value = 0.0f;
  const device bfloat* v = qkv + 2 * width + head * head_dim + 4 * group;
  for (uint key = split; key < rows; key += splits) {
    value += weights[key] * float4(*reinterpret_cast<const device bfloat4*>(v + key * 3 * width));
  }
  for (uint offset = quads; offset < LANES; offset *= 2) {
    value += simd_shuffle_down(value, ushort(offset));
  }
  if (split == 0) {
    *reinterpret_cast<device bfloat4*>(out + 4 * group) = bfloat4(value / total);
  }
}
