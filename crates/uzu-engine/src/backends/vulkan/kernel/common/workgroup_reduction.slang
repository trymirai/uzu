import definitions;

// Reductions at any subgroup width: the subgroup operation, then at most one shared stage across subgroups. Every
// invocation of a workgroup of at most THREADS must call each reduction; all of a group read the same result.

// An associative, commutative combination: `combine` of two values, `subgroup` over the active lanes of a subgroup.
public interface IReduction {
  associatedtype Value;
  static Value combine(Value a, Value b);
  static Value subgroup(Value value);
}

public struct Sum<T : __BuiltinArithmeticType> : IReduction {
  public typealias Value = T;
  public static T combine(T a, T b) { return a + b; }
  public static T subgroup(T value) { return WaveActiveSum(value); }
}

public struct VectorSum<let N : int> : IReduction {
  public typealias Value = vector<float, N>;
  public static Value combine(Value a, Value b) { return a + b; }
  public static Value subgroup(Value value) { return WaveActiveSum(value); }
}

public struct Max<T : __BuiltinArithmeticType> : IReduction {
  public typealias Value = T;
  public static T combine(T a, T b) { return max(a, b); }
  public static T subgroup(T value) { return WaveActiveMax(value); }
}

// One entry per subgroup, shared by every reduction of the same value type and bound.
public struct WorkgroupScratch<T, let THREADS : uint> {
  public static groupshared T data[THREADS];
}

// R over the workgroup: R.subgroup, one shared entry per subgroup, then subgroup 0 folds the entries.
public R.Value workgroup_reduce<R : IReduction, let THREADS : uint>(R.Value value) {
  static_assert(THREADS <= 1024, "a workgroup bound");
  typealias S = WorkgroupScratch<R.Value, THREADS>;
  if (WaveGetLaneCount() > 64) {
    // Subgroup operations of wider subgroups cost more than a tree of shared stages.
    let index = subgroup_invocation();
    let threads = WaveGetNumWaves() * WaveGetLaneCount();
    var span = 1u;
    while (span < threads) {
      span <<= 1;
    }
    S.data[index] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = span / 2; stride > 0; stride >>= 1) {
      if (index < stride && index + stride < threads) {
        S.data[index] = R.combine(S.data[index], S.data[index + stride]);
      }
      GroupMemoryBarrierWithGroupSync();
    }
    let total = S.data[0];
    GroupMemoryBarrierWithGroupSync();
    return total;
  }
  let partial = R.subgroup(value);
  if (WaveIsFirstLane()) {
    S.data[subgroup_id()] = partial;
  }
  GroupMemoryBarrierWithGroupSync();
  if (subgroup_id() == 0) {
    // Active lanes stride over every entry, even when there are more subgroups than lanes.
    let lane = WavePrefixCountBits(true);
    let lanes = WaveActiveCountBits(true);
    if (lane < WaveGetNumWaves()) {
      var total = S.data[lane];
      for (uint entry = lane + lanes; entry < WaveGetNumWaves(); entry += lanes) {
        total = R.combine(total, S.data[entry]);
      }
      total = R.subgroup(total);
      if (lane == 0) {
        S.data[0] = total;
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();
  let total = S.data[0];
  // Every invocation has read the result before another reduction reuses the scratch.
  GroupMemoryBarrierWithGroupSync();
  return total;
}

// R over the aligned run of `size` invocations in subgroup_invocation order holding this one, `size` a uniform power of
// two: R.subgroup for a whole subgroup, shuffle-xor butterflies for a narrower run, else one shared stage.
public R.Value segment_reduce<R : IReduction, let THREADS : uint>(R.Value value, uint size) {
  static_assert(THREADS <= 1024, "a workgroup bound");
  let width = WaveGetLaneCount();
  if (width > 64) {
    // As workgroup_reduce: a tree of shared stages within each run.
    typealias S = WorkgroupScratch<R.Value, THREADS>;
    let index = subgroup_invocation();
    S.data[index] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = size / 2; stride > 0; stride >>= 1) {
      if ((index & (size - 1)) < stride) {
        S.data[index] = R.combine(S.data[index], S.data[index + stride]);
      }
      GroupMemoryBarrierWithGroupSync();
    }
    let total = S.data[index & ~(size - 1)];
    GroupMemoryBarrierWithGroupSync();
    return total;
  }
  if (size == width) {
    return R.subgroup(value);
  }
  if (size < width) {
    var reduced = value;
    for (uint offset = size / 2; offset > 0; offset /= 2) {
      reduced = R.combine(reduced, shuffle_xor(reduced, offset));
    }
    return reduced;
  }
  typealias S = WorkgroupScratch<R.Value, THREADS>;
  let partial = R.subgroup(value);
  if (WaveIsFirstLane()) {
    S.data[subgroup_id()] = partial;
  }
  GroupMemoryBarrierWithGroupSync();
  let subgroups = size / width;
  let first = subgroup_id() & ~(subgroups - 1);
  var total = S.data[first];
  for (uint entry = 1; entry < subgroups; entry++) {
    total = R.combine(total, S.data[first + entry]);
  }
  GroupMemoryBarrierWithGroupSync();
  return total;
}

// The exclusive prefix sum of `value` over the workgroup in subgroup_invocation order, and in `total` the sum.
public T workgroup_prefix_sum<T : __BuiltinArithmeticType, let THREADS : uint>(T value, out T total) {
  static_assert(THREADS <= 1024, "a workgroup bound");
  typealias S = WorkgroupScratch<T, THREADS>;
  let prefix = WavePrefixSum(value);
  let subgroup_total = WaveActiveSum(value);
  if (WaveIsFirstLane()) {
    S.data[subgroup_id()] = subgroup_total;
  }
  GroupMemoryBarrierWithGroupSync();
  var below = T(0);
  for (uint subgroup = 0; subgroup < subgroup_id(); subgroup++) {
    below += S.data[subgroup];
  }
  total = below;
  for (uint subgroup = subgroup_id(); subgroup < WaveGetNumWaves(); subgroup++) {
    total += S.data[subgroup];
  }
  GroupMemoryBarrierWithGroupSync();
  return below + prefix;
}

// 1 / sqrt(sum / count + epsilon) over the workgroup from each invocation's partial sum of squares.
public float rms_inverse<let THREADS : uint>(float partial, uint count, float epsilon) {
  return rsqrt(workgroup_reduce<Sum<float>, THREADS>(partial) / float(count) + epsilon);
}
