import definitions;
import common.activation;
import common.storage_address;
import generated.activation_type;

// The CPU SSDUpdate of Mamba2, one decode step of the state-space recurrence. Each (batch, head, head element) is an
// independent invocation that owns its y element and its state row: it updates the row in ascending state order, reading
// each element before writing it, and folds the new state into y. Every product and sum is one IEEE FP32 operation
// (exact_product, ordered_sum) rounded to T, as the CPU's T arithmetic: next = T(T(s decay) + T(b x)), the accumulator
// from +0 over T(next c) in order, then T(d x), then the gate. dt, decay and the gate are rounded to T. The last stride
// of B/C and of the state are ignored, as on the CPU. Element offsets are 64-bit words, and a pointer to a B, C or state
// row is formed only where that row is read or written. Workgroups stride over their dimension by the actual
// NumWorkgroups, which the host clamps to the device limits, advancing only while more than one step remains.

static const uint SSD_UPDATE_THREADS = 64;

// T(a b) and T(a + b), each one IEEE FP32 operation with subnormals kept, never fused.
T ssd_product<T: IStorageFloat>(T a, T b) {
  return T.fromFloat(exact_product(a.toFloat(), b.toFloat()));
}

T ssd_sum<T: IStorageFloat>(T a, T b) {
  return T.fromFloat(ordered_sum(a.toFloat(), b.toFloat()));
}

// exp(-dt) for dt = softplus, which is +0 or more, +inf or NaN. GLSL Exp does not guarantee its result for special
// inputs, so these bit branches decide them: NaN propagates and +inf gives +0, the CPU's expf(-inf).
public float ssd_decay(float dt) {
  let bits = asuint(dt);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return dt;
  }
  if (bits == 0x7f800000u) {
    return 0.0;
  }
  return exp(-dt);
}

// The element index.x strides.x + index.y strides.y + index.z strides.z past `base`, one 64-bit term at a time.
Ptr<T, access> ssd_element<T, Access access>(Ptr<T, access> base, uint3 index, uint3 strides) {
  let row = element_pointer(base, element_index(index.x, strides.x, 0));
  return element_pointer(element_pointer(row, element_index(index.y, strides.y, 0)), element_index(index.z, strides.z, 0));
}

// A state element: next_state's in place, where the original state is absent and never addressed, otherwise the
// original state's.
T ssd_state<T: IStorageFloat, let IN_PLACE : bool>(
  Ptr<T, Access.Read> state_row,
  Ptr<T, Access.ReadWrite> next_row,
  uint i,
) {
  if (IN_PLACE) {
    return element_pointer(next_row, uint2(i, 0))[0];
  }
  return element_pointer(state_row, uint2(i, 0))[0];
}

// One (batch, head, head element), reading each state element before writing its new value.
void ssd_update_item<T: IStorageFloat, let IN_PLACE : bool>(
  Ptr<T, Access.Read> x,
  Ptr<T, Access.Read> dt_raw,
  Ptr<T, Access.Read> b,
  Ptr<T, Access.Read> c,
  Ptr<T, Access.Read> d,
  Ptr<T, Access.Read> z,
  Ptr<T, Access.Read> state,
  Ptr<T, Access.ReadWrite> y,
  Ptr<T, Access.ReadWrite> next_state,
  uint group_size,
  uint state_size,
  uint3 x_strides,
  uint3 dt_strides,
  uint3 cb_strides,
  uint3 state_strides,
  uint3 item,
) {
  let x_value = ssd_element(x, item, x_strides)[0];
  let dt = T.fromFloat(activate(ssd_element(dt_raw, item, dt_strides)[0].toFloat(), ActivationType.SOFTPLUS));
  let decay = T.fromFloat(ssd_decay(dt.toFloat()));
  let gate = T.fromFloat(activate(ssd_element(z, item, x_strides)[0].toFloat(), ActivationType.SILU));
  var acc = T.fromFloat(0.0);
  if (state_size > 0) {
    let group = uint3(item.x, item.y / group_size, 0);
    let b_row = ssd_element(b, group, cb_strides);
    let c_row = ssd_element(c, group, cb_strides);
    let next_row = ssd_element(next_state, item, state_strides);
    var state_row = state;
    if (!IN_PLACE) {
      state_row = ssd_element(state, item, state_strides);
    }
    for (uint i = 0; i < state_size; i++) {
      let s = ssd_state<T, IN_PLACE>(state_row, next_row, i);
      let next = ssd_sum(ssd_product(s, decay), ssd_product(element_pointer(b_row, uint2(i, 0))[0], x_value));
      element_pointer(next_row, uint2(i, 0))[0] = next;
      acc = ssd_sum(acc, ssd_product(next, element_pointer(c_row, uint2(i, 0))[0]));
    }
  }
  acc = ssd_sum(acc, ssd_product(element_pointer(d, uint2(item.y, 0))[0], x_value));
  ssd_element(y, item, x_strides)[0] = ssd_product(acc, gate);
}

[[Kernel]] [[Public]]
[[Precondition("encode", "group_size > 0 || b_size == 0 || h_size == 0 || dh_size == 0")]]
public func SSDUpdate<T: IStorageFloat>(
  Ptr<T, Access.Read> x,
  Ptr<T, Access.Read> dt_raw,
  Ptr<T, Access.Read> b,
  Ptr<T, Access.Read> c,
  Ptr<T, Access.Read> d,
  Ptr<T, Access.Read> z,
  [[Optional("!state_in_place")]] Ptr<T, Access.Read> state,
  Ptr<T, Access.ReadWrite> y,
  Ptr<T, Access.ReadWrite> next_state,
  uint group_size,
  uint state_size,
  [[HostSlice(3)]] Ptr<uint, Access.Read> x_strides,
  [[HostSlice(2)]] Ptr<uint, Access.Read> dt_strides,
  [[HostSlice(3)]] Ptr<uint, Access.Read> cb_strides,
  [[HostSlice(4)]] Ptr<uint, Access.Read> state_strides,
  uint b_size,
  uint h_size,
  uint dh_size,
  [[Groups("dh_size.div_ceil(64).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0])")]] uint dh_group,
  [[Groups("h_size.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint h_group,
  [[Groups("b_size.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[2])")]] uint b_group,
  [[Threads("64")]] uint thread_idx,
  [[Specialize]] bool state_in_place,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let xs = uint3(x_strides[0], x_strides[1], x_strides[2]);
  let dts = uint3(dt_strides[0], dt_strides[1], 0);
  let cbs = uint3(cb_strides[0], cb_strides[1], 0);
  let ss = uint3(state_strides[0], state_strides[1], state_strides[2]);
  let dh_groups = dh_size / SSD_UPDATE_THREADS + (dh_size % SSD_UPDATE_THREADS != 0 ? 1u : 0u);
  var batch = b_group;
  while (batch < b_size) {
    var head = h_group;
    while (head < h_size) {
      var group = dh_group;
      while (group < dh_groups) {
        let element = group * SSD_UPDATE_THREADS + thread_idx;
        if (element < dh_size) {
          let item = uint3(batch, head, element);
          if (state_in_place) {
            ssd_update_item<T, true>(
              x, dt_raw, b, c, d, z, state, y, next_state, group_size, state_size, xs, dts, cbs, ss, item
            );
          } else {
            ssd_update_item<T, false>(
              x, dt_raw, b, c, d, z, state, y, next_state, group_size, state_size, xs, dts, cbs, ss, item
            );
          }
        }
        if (dh_groups - group <= groups.x) {
          break;
        }
        group += groups.x;
      }
      if (h_size - head <= groups.y) {
        break;
      }
      head += groups.y;
    }
    if (b_size - batch <= groups.z) {
      break;
    }
    batch += groups.z;
  }
}
