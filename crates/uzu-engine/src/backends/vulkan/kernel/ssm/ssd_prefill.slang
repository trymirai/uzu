import definitions;
import common.activation;
import common.storage_address;
import generated.activation_type;
import ssm.ssd_update;

// The CPU SSDPrefill of Mamba2: the SSD recurrence over suffix_len tokens Q for every head h and head element e. x, z
// and y are [Q, H, Dh] at x_strides, dt_raw [Q, H] at dt_strides, B and C [Q, groups, N] at cb_strides with group
// h / max(group_size, 1), d [H], and the state [H, Dh, N] at state_strides, every stride used. One invocation owns
// (h, e): it widens its state row to a private FP32 row once, keeps it in FP32 across all tokens and stores it as T once
// at the end. Per token, each an IEEE FP32 operation never fused (exact_product, ordered_sum): x widens, the gate is
// SiLU rounded to T and widened, dt = softplus and decay = exp(-dt) stay FP32 (ssd_decay, SSDUpdate's), s_i = s_i decay +
// B_i x for i ascending, the dot from +0 over s_i C_i, then y = T((dot + d x) gate). With N 0, dt_raw, B, C and the
// state are never addressed. Element offsets are 64-bit words. Workgroups stride over Dh and H by the actual
// NumWorkgroups, which the host clamps to the device limits, advancing only while more than one step remains.

static const uint SSD_PREFILL_THREADS = 64;

// The generic kernel's state size, owned by this module: it sizes the private row, max(N, 1) elements without select.
[[SpecializationConstant]] const uint __dsl_SSDPrefill_state_size = 64;
static const uint SSD_PREFILL_CAPACITY = __dsl_SSDPrefill_state_size | (1u - ((__dsl_SSDPrefill_state_size | (0u - __dsl_SSDPrefill_state_size)) >> 31));

// Every (h, e) of the grid, its private row 64 elements for SSDPrefill64 and SSD_PREFILL_CAPACITY otherwise, of which
// the first state_size are live.
void ssd_prefill_grid<T: IStorageFloat, let SPECIAL64 : bool>(
  Ptr<T, Access.Read> x,
  Ptr<T, Access.Read> dt_raw,
  Ptr<T, Access.Read> b,
  Ptr<T, Access.Read> c,
  Ptr<T, Access.Read> d,
  Ptr<T, Access.Read> z,
  Ptr<T, Access.ReadWrite> state,
  Ptr<T, Access.ReadWrite> y,
  uint suffix_len,
  uint group_size,
  uint state_size,
  Ptr<uint, Access.Read> x_strides,
  Ptr<uint, Access.Read> dt_strides,
  Ptr<uint, Access.Read> cb_strides,
  Ptr<uint, Access.Read> state_strides,
  uint num_heads,
  uint head_dim,
  uint dh_group,
  uint h_group,
  uint thread_idx,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let dh_groups = head_dim / SSD_PREFILL_THREADS + (head_dim % SSD_PREFILL_THREADS != 0 ? 1u : 0u);
  var head = h_group;
  while (head < num_heads) {
    var group = dh_group;
    while (group < dh_groups) {
      let element = group * SSD_PREFILL_THREADS + thread_idx;
      if (element < head_dim) {
        // x, z and y of (head, element), and its state row, each formed one 64-bit term at a time.
        let xs = element_index(element, x_strides[2], 0);
        let x_row = element_pointer(element_pointer(x, element_index(head, x_strides[1], 0)), xs);
        let z_row = element_pointer(element_pointer(z, element_index(head, x_strides[1], 0)), xs);
        let y_row = element_pointer(element_pointer(y, element_index(head, x_strides[1], 0)), xs);
        let skip = element_pointer(d, uint2(head, 0))[0].toFloat();
        let cb_group = head / max(group_size, 1u);
        float row[SPECIAL64 ? 64u : SSD_PREFILL_CAPACITY];
        var state_row = state;
        if (state_size > 0) {
          state_row = element_pointer(element_pointer(state, element_index(head, state_strides[0], 0)), element_index(element, state_strides[1], 0));
          for (uint i = 0; i < state_size; i++) {
            row[i] = element_pointer(state_row, element_index(i, state_strides[2], 0))[0].toFloat();
          }
        }
        for (uint t = 0; t < suffix_len; t++) {
          let token = element_index(t, x_strides[0], 0);
          let this_x = element_pointer(x_row, token)[0].toFloat();
          let gate = T.fromFloat(activate(element_pointer(z_row, token)[0].toFloat(), ActivationType.SILU)).toFloat();
          var dot = 0.0;
          if (state_size > 0) {
            let dt_row = element_pointer(dt_raw, element_index(t, dt_strides[0], 0));
            let dt = activate(element_pointer(dt_row, element_index(head, dt_strides[1], 0))[0].toFloat(), ActivationType.SOFTPLUS);
            let decay = ssd_decay(dt);
            let cb_token = element_index(t, cb_strides[0], 0);
            let b_row = element_pointer(element_pointer(b, cb_token), element_index(cb_group, cb_strides[1], 0));
            let c_row = element_pointer(element_pointer(c, cb_token), element_index(cb_group, cb_strides[1], 0));
            for (uint i = 0; i < state_size; i++) {
              let cb = element_index(i, cb_strides[2], 0);
              row[i] = ordered_sum(exact_product(row[i], decay), exact_product(element_pointer(b_row, cb)[0].toFloat(), this_x));
              dot = ordered_sum(dot, exact_product(row[i], element_pointer(c_row, cb)[0].toFloat()));
            }
          }
          element_pointer(y_row, token)[0] = T.fromFloat(exact_product(ordered_sum(dot, exact_product(skip, this_x)), gate));
          if (suffix_len - t <= 1) {
            break;
          }
        }
        if (state_size > 0) {
          for (uint i = 0; i < state_size; i++) {
            element_pointer(state_row, element_index(i, state_strides[2], 0))[0] = T.fromFloat(row[i]);
          }
        }
      }
      if (dh_groups - group <= groups.x) {
        break;
      }
      group += groups.x;
    }
    if (num_heads - head <= groups.y) {
      break;
    }
    head += groups.y;
  }
}

// SSDPrefill for a state of exactly 64 elements, its private row a literal 64. Declared first, in the CPU kernels' order.
[[Kernel]] [[Public]]
[[Precondition("encode", "suffix_len == 0 || num_heads == 0 || head_dim == 0 || state_size == 64")]]
func SSDPrefill64<T: IStorageFloat>(
  Ptr<T, Access.Read> x,
  Ptr<T, Access.Read> dt_raw,
  Ptr<T, Access.Read> b,
  Ptr<T, Access.Read> c,
  Ptr<T, Access.Read> d,
  Ptr<T, Access.Read> z,
  Ptr<T, Access.ReadWrite> state,
  Ptr<T, Access.ReadWrite> y,
  uint suffix_len,
  uint group_size,
  uint state_size,
  [[HostSlice(3)]] Ptr<uint, Access.Read> x_strides,
  [[HostSlice(2)]] Ptr<uint, Access.Read> dt_strides,
  [[HostSlice(3)]] Ptr<uint, Access.Read> cb_strides,
  [[HostSlice(3)]] Ptr<uint, Access.Read> state_strides,
  uint num_heads,
  uint head_dim,
  [[Groups("if suffix_len == 0 { 0 } else { head_dim.div_ceil(64).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint dh_group,
  [[Groups("num_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint h_group,
  [[Threads("64")]] uint thread_idx,
) {
  ssd_prefill_grid<T, true>(
    x, dt_raw, b, c, d, z, state, y, suffix_len, group_size, state_size, x_strides, dt_strides, cb_strides, state_strides,
    num_heads, head_dim, dh_group, h_group, thread_idx
  );
}

[[Kernel]] [[Public]]
func SSDPrefill<T: IStorageFloat>(
  Ptr<T, Access.Read> x,
  Ptr<T, Access.Read> dt_raw,
  Ptr<T, Access.Read> b,
  Ptr<T, Access.Read> c,
  Ptr<T, Access.Read> d,
  Ptr<T, Access.Read> z,
  Ptr<T, Access.ReadWrite> state,
  Ptr<T, Access.ReadWrite> y,
  uint suffix_len,
  uint group_size,
  [[Specialize]] uint state_size,
  [[HostSlice(3)]] Ptr<uint, Access.Read> x_strides,
  [[HostSlice(2)]] Ptr<uint, Access.Read> dt_strides,
  [[HostSlice(3)]] Ptr<uint, Access.Read> cb_strides,
  [[HostSlice(3)]] Ptr<uint, Access.Read> state_strides,
  uint num_heads,
  uint head_dim,
  [[Groups("if suffix_len == 0 { 0 } else { head_dim.div_ceil(64).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint dh_group,
  [[Groups("num_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint h_group,
  [[Threads("64")]] uint thread_idx,
) {
  ssd_prefill_grid<T, false>(
    x, dt_raw, b, c, d, z, state, y, suffix_len, group_size, state_size, x_strides, dt_strides, cb_strides, state_strides,
    num_heads, head_dim, dh_group, h_group, thread_idx
  );
}
