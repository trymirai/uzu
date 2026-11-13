import definitions;
import common.workgroup_reduction;
import generated.confidence_head;

public static const uint HEAD_NONE = 0xffffffffu;
public static const float HEAD_LOG_PROB_FLOOR = -30.0;
public static const uint HEAD_RELATION_ABSENT = RELATIONS + 1;

public float head_log_sum_exp<let THREADS : uint>(float value) {
  let top = workgroup_reduce<Max<float>, THREADS>(value);
  return top + log(workgroup_reduce<Sum<float>, THREADS>(value == NEGATIVE_INFINITY ? 0.0 : exp(value - top)));
}
