import definitions;
import common.workgroup_reduction;
import confidence_head.head;
import generated.confidence_head;

// Attention of each present tree row over the present rows, a workgroup per (row, head): a score is the scaled
// query-key dot plus the head's bias for the pair's relation, none for a row with itself. Absent rows output zero.

static const uint TREE_ATTENTION_THREADS = 32;

groupshared float tree_attention_weights[TREE_ROWS_MAX];

[[Kernel]] [[Public]]
[[Precondition("encode", "rows <= crate::backends::common::gpu_types::confidence_head::TREE_ROWS_MAX")]]
func ConfidenceHeadTreeAttention(
  Ptr<bf16, Access.Read> qkv,
  Ptr<uint, Access.Read> relations,
  Ptr<bf16, Access.Read> relation_bias,
  Ptr<bf16, Access.ReadWrite> output,
  uint rows,
  [[Groups("rows * num_heads")]] uint index,
  [[Threads("32")]] uint lane,
  [[Specialize]] uint num_heads,
  [[Specialize]] uint head_dim,
) {
  let query = index / num_heads;
  let head = index % num_heads;
  let width = num_heads * head_dim;
  let relation = relations + query * rows;
  let destination = output + query * width + head * head_dim;
  let invocation = subgroup_invocation();
  if (relation[query] == HEAD_RELATION_ABSENT) {
    for (uint unit = invocation; unit < head_dim; unit += TREE_ATTENTION_THREADS) {
      destination[unit] = bf16.fromFloat(0.0);
    }
    return;
  }
  let q = qkv + query * 3 * width + head * head_dim;
  let scale = rsqrt(float(head_dim));
  var local_top = NEGATIVE_INFINITY;
  for (uint key = invocation; key < rows; key += TREE_ATTENTION_THREADS) {
    let code = relation[key];
    var score = NEGATIVE_INFINITY;
    if (code != HEAD_RELATION_ABSENT) {
      let k = qkv + key * 3 * width + width + head * head_dim;
      var product = 0.0;
      for (uint unit = 0; unit < head_dim; unit++) {
        product = fma(q[unit].toFloat(), k[unit].toFloat(), product);
      }
      score = product * scale + (code == 0 ? 0.0 : relation_bias[head * RELATIONS + code - 1].toFloat());
    }
    tree_attention_weights[key] = score;
    local_top = max(local_top, score);
  }
  let top = workgroup_reduce<Max<float>, TREE_ATTENTION_THREADS>(local_top);
  var local_total = 0.0;
  for (uint key = invocation; key < rows; key += TREE_ATTENTION_THREADS) {
    let weight = exp(tree_attention_weights[key] - top);
    tree_attention_weights[key] = weight;
    local_total += weight;
  }
  // The reduction's barriers also publish every key's weight.
  let total = workgroup_reduce<Sum<float>, TREE_ATTENTION_THREADS>(local_total);
  let v = qkv + 2 * width + head * head_dim;
  for (uint unit = invocation; unit < head_dim; unit += TREE_ATTENTION_THREADS) {
    var value = 0.0;
    for (uint key = 0; key < rows; key++) {
      value = fma(tree_attention_weights[key], v[key * 3 * width + unit].toFloat(), value);
    }
    destination[unit] = bf16.fromFloat(value / total);
  }
}
