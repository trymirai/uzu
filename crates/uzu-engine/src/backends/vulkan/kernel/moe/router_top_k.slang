import definitions;
import common.ranking;
import common.workgroup_reduction;

// Each token's top_k experts by logit, descending, a tie to the lower expert and NaN below every logit, with the
// softmax of their logits. A workgroup ranks one token: each invocation holds every MOE_ROUTER_THREADS-th expert's key,
// and each pick is the workgroup's best key not yet picked.

static const uint MOE_ROUTER_THREADS = 32;
static const uint MOE_ROUTER_KEYS = 8;
static const uint MOE_ROUTER_MAX_TOP_K = 8;

// The f32::total_cmp key of a logit, NaN as negative infinity; every key is above 0.
uint moe_router_key(float logit) {
  let bits = asuint(logit);
  return float_order_key((bits & 0x7fffffffu) > 0x7f800000u ? asuint(NEGATIVE_INFINITY) : bits);
}

[[Variants("T", "float")]]
[[Kernel]] [[Public]]
[[Precondition("new", "MAX_TOP_K == 8")]]
[[Precondition("encode", "num_experts <= 256 && top_k > 0 && top_k <= MAX_TOP_K && top_k <= num_experts")]]
func MoeRouterTopK<T: IStorageFloat>(
  Ptr<T, Access.Read> logits,
  Ptr<uint, Access.ReadWrite> expert_ids,
  Ptr<float, Access.ReadWrite> expert_weights,
  uint num_experts,
  uint top_k,
  uint batch_size,
  [[Groups("batch_size")]] uint token,
  [[Threads("32")]] uint lane,
  [[Specialize]] uint MAX_TOP_K,
) {
  let invocation = subgroup_invocation();
  uint keys[MOE_ROUTER_KEYS];
  for (uint slot = 0; slot < MOE_ROUTER_KEYS; slot++) {
    let expert = invocation + slot * MOE_ROUTER_THREADS;
    keys[slot] = expert < num_experts ? moe_router_key(logits[token * num_experts + expert].toFloat()) : 0;
  }

  float picked[MOE_ROUTER_MAX_TOP_K];
  for (uint pick = 0; pick < top_k; pick++) {
    // WeaverRank orders by descending key, then ascending expert.
    var best = uint4(0, 0xffffffffu, 0, 0);
    for (uint slot = 0; slot < MOE_ROUTER_KEYS; slot++) {
      let candidate = uint4(keys[slot], invocation + slot * MOE_ROUTER_THREADS, 0, 0);
      best = WeaverRank.combine(best, candidate);
    }
    let chosen = workgroup_reduce<WeaverRank, MOE_ROUTER_THREADS>(best);
    if (chosen.y % MOE_ROUTER_THREADS == invocation) {
      keys[chosen.y / MOE_ROUTER_THREADS] = 0;
    }
    picked[pick] = asfloat(float_order_bits(chosen.x));
    if (invocation == 0) {
      expert_ids[token * top_k + pick] = chosen.y;
    }
  }

  if (invocation == 0) {
    let largest = picked[0];
    var sum = 0.0;
    for (uint pick = 0; pick < top_k; pick++) {
      picked[pick] = largest == NEGATIVE_INFINITY ? 1.0 / float(top_k) : exp(picked[pick] - largest);
      sum += picked[pick];
    }
    let inverse_sum = sum > 0.0 ? 1.0 / sum : 0.0;
    for (uint pick = 0; pick < top_k; pick++) {
      expert_weights[token * top_k + pick] = picked[pick] * inverse_sum;
    }
  }
}
