import definitions;

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

public struct WorkgroupScratch<T, let THREADS : uint> {
  public static groupshared T data[THREADS];
}

public R.Value workgroup_reduce<R : IReduction, let THREADS : uint>(R.Value value) {
  static_assert(THREADS <= 1024, "a workgroup bound");
  typealias S = WorkgroupScratch<R.Value, THREADS>;
  if (WaveGetLaneCount() > 64) {
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
  GroupMemoryBarrierWithGroupSync();
  return total;
}

public R.Value segment_reduce<R : IReduction, let THREADS : uint>(R.Value value, uint size) {
  static_assert(THREADS <= 1024, "a workgroup bound");
  let width = WaveGetLaneCount();
  if (width > 64) {
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

public T workgroup_prefix_sum<T : __BuiltinArithmeticType, let THREADS : uint>(T value) {
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
  GroupMemoryBarrierWithGroupSync();
  return below + prefix;
}

public float rms_inverse<let THREADS : uint>(float partial, uint count, float epsilon) {
  return rsqrt(workgroup_reduce<Sum<float>, THREADS>(partial) / float(count) + epsilon);
}
