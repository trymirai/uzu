import definitions;
import common.activation;
import common.storage_address;
import gdn.prefill_prep;

// CPU-ordered FP32 forward substitution of (I + A) U = beta (v - e^prefix kh0) over 16-token blocks, never fused.
// Each invocation owns u[b H + hv, 0..N, dv] and re-reads only its own earlier writes; inputs must not alias u.
// The exponential is always evaluated and every valid Ainv product, including upper and zero entries, is included.
// Addresses are termwise 64-bit; clamped workgroups stride over value-dim blocks and heads without uint wrap.

static const uint TREE_UPDATE_SOLVE_THREADS = 64;
static const uint TREE_UPDATE_SOLVE_BLOCK = 16;

[[Kernel]] [[Public]]
[[Precondition("new", "BV == 16 || BV == 32")]]
func TreeUpdateSolve<T: IModelFloat>(
  [[Optional("use_h0")]] Ptr<float, Access.Read> kh0,
  Ptr<T, Access.Read> v,
  Ptr<float, Access.Read> prefix,
  Ptr<float, Access.Read> beta,
  Ptr<float, Access.Read> a_packed,
  Ptr<float, Access.Read> a_inv,
  [[Optional("use_h0")]] Ptr<int, Access.Read> h0_idx,
  Ptr<float, Access.ReadWrite> u,
  uint batch_size,
  uint tree_size,
  uint value_heads,
  uint head_v_dim,
  [[Groups("if batch_size == 0 || tree_size == 0 || value_heads == 0 || head_v_dim == 0 { 0 } else { head_v_dim.div_ceil(64).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint dv_group,
  [[Groups("if batch_size == 0 || tree_size == 0 || value_heads == 0 || head_v_dim == 0 { 0 } else { value_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1]) }")]] uint head_group,
  [[Threads("64")]] uint thread_idx,
  [[Specialize]] uint BV,
  [[Specialize]] bool use_h0,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let blocks = tree_size / TREE_UPDATE_SOLVE_BLOCK + (tree_size % TREE_UPDATE_SOLVE_BLOCK != 0 ? 1u : 0u);
  let pairs = blocks / 2 + blocks % 2;
  let dv_blocks = head_v_dim / TREE_UPDATE_SOLVE_THREADS + (head_v_dim % TREE_UPDATE_SOLVE_THREADS != 0 ? 1u : 0u);
  var hv = head_group;
  while (hv < value_heads) {
    var dv_block = dv_group;
    while (dv_block < dv_blocks) {
      for (uint b = 0; b < batch_size; b++) {
        var slot = -1;
        if (use_h0) {
          slot = element_pointer(h0_idx, uint2(b, 0))[0];
        }
        let head = element_index(b, value_heads, hv);
        let tiles = element_pointer(a_packed, byte_offset(byte_offset(byte_offset(head, blocks), pairs), 512));
        let inverses = element_pointer(a_inv, byte_offset(byte_offset(head, blocks), 256));
        let column = element_pointer(u, byte_offset(byte_offset(head, tree_size), head_v_dim));
        for (uint block = 0; block < blocks; block++) {
          let first = block * TREE_UPDATE_SOLVE_BLOCK;
          let rows = min(TREE_UPDATE_SOLVE_BLOCK, tree_size - first);
          if (thread_idx < head_v_dim - dv_block * TREE_UPDATE_SOLVE_THREADS) {
            let dv = dv_block * TREE_UPDATE_SOLVE_THREADS + thread_idx;
            float acc[TREE_UPDATE_SOLVE_BLOCK];
            for (uint lt = 0; lt < rows; lt++) {
              let lane = add_address(byte_offset(element_index(b, tree_size, first + lt), value_heads), uint2(hv, 0));
              let element = add_address(byte_offset(lane, head_v_dim), uint2(dv, 0));
              var kh = 0.0;
              if (slot >= 0) {
                kh = element_pointer(kh0, element)[0];
              }
              let e = delta_net_exp(element_pointer(prefix, lane)[0]);
              let rhs = ordered_sum(element_pointer(v, element)[0].toFloat(), -exact_product(e, kh));
              acc[lt] = exact_product(element_pointer(beta, lane)[0], rhs);
            }
            for (uint prev = 0; prev < first; prev++) {
              let tile = element_pointer(tiles, byte_offset(element_index(block, pairs, prev / 32), 512));
              let u_prev = element_pointer(column, element_index(prev, head_v_dim, dv))[0];
              for (uint lt = 0; lt < rows; lt++) {
                let a = element_pointer(tile, element_index(lt, 32, prev % 32))[0];
                acc[lt] = ordered_sum(acc[lt], -exact_product(a, u_prev));
              }
            }
            let inverse = element_pointer(inverses, byte_offset(uint2(block, 0), 256));
            for (uint lt = 0; lt < rows; lt++) {
              var s = 0.0;
              for (uint lp = 0; lp < rows; lp++) {
                let ainv = element_pointer(inverse, element_index(lt, TREE_UPDATE_SOLVE_BLOCK, lp))[0];
                s = ordered_sum(s, exact_product(ainv, acc[lp]));
              }
              element_pointer(column, element_index(first + lt, head_v_dim, dv))[0] = s;
            }
          }
        }
      }
      if (dv_blocks - dv_block <= groups.x) {
        break;
      }
      dv_block += groups.x;
    }
    if (value_heads - hv <= groups.y) {
      break;
    }
    hv += groups.y;
  }
}
