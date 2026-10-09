import definitions;
import common.activation;
import common.storage_address;

// The CPU split_in_proj: copies the stored T bits of each in_proj row's first conv_dim columns and of the num_heads after
// the next inner_dim, whose FP32 sums with z_bias are rounded once to T. A smaller total_dim truncates the segments;
// later columns are never loaded. Routing by subtraction and 64-bit element offsets keep the CPU's indexing.
// Workgroups stride over 256-column groups and rows by the actual NumWorkgroups, which the host clamps to the device
// limits, advancing only while more than one step remains so no counter passes its bound.

static const uint SPLIT_IN_PROJ_THREADS = 256;

// Column `column` of every `step`-th row from `row` to element `local` of `stride`-element output rows: the stored value,
// or with BIAS T(x + bias).
void split_rows<T: IStorageFloat, let BIAS : bool>(
  Ptr<T, Access.Read> input,
  Ptr<T, Access.ReadWrite> output,
  float bias,
  uint total_dim,
  uint column,
  uint stride,
  uint local,
  uint row,
  uint step,
  uint suffix_length,
) {
  while (row < suffix_length) {
    let source = element_pointer(input, element_index(row, total_dim, column));
    let target = element_pointer(output, element_index(row, stride, local));
    if (BIAS) {
      target[0] = T.fromFloat(exact_sum(source[0].toFloat(), bias, false));
    } else {
      target[0] = source[0];
    }
    if (suffix_length - row <= step) {
      return;
    }
    row += step;
  }
}

[[Kernel]] [[Public]] func SplitInProj<T: IStorageFloat>(
  Ptr<T, Access.Read> input,
  Ptr<T, Access.ReadWrite> conv_out,
  Ptr<T, Access.ReadWrite> z_out,
  Ptr<T, Access.ReadWrite> dt_out,
  Ptr<T, Access.Read> z_bias,
  uint suffix_length,
  uint total_dim,
  uint conv_dim,
  uint inner_dim,
  uint num_heads,
  [[Groups("total_dim.min(conv_dim.saturating_add(inner_dim).saturating_add(num_heads)).div_ceil(256).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0])")]] uint column_group,
  [[Groups("suffix_length.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint row_group,
  [[Threads("256")]] uint thread_idx,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  // min(total_dim, conv_dim + inner_dim + num_heads) with saturated sums.
  let z_end = conv_dim + min(inner_dim, 0xffffffffu - conv_dim);
  let columns = min(total_dim, z_end + min(num_heads, 0xffffffffu - z_end));
  let column_groups = columns / SPLIT_IN_PROJ_THREADS + (columns % SPLIT_IN_PROJ_THREADS != 0 ? 1u : 0u);
  var group = column_group;
  while (group < column_groups) {
    let column = group * SPLIT_IN_PROJ_THREADS + thread_idx;
    if (column < columns) {
      if (column < conv_dim) {
        split_rows<T, false>(input, conv_out, 0.0, total_dim, column, conv_dim, column, row_group, groups.y, suffix_length);
      } else if (column - conv_dim < inner_dim) {
        let local = column - conv_dim;
        let bias = element_pointer(z_bias, uint2(local, 0))[0].toFloat();
        split_rows<T, true>(input, z_out, bias, total_dim, column, inner_dim, local, row_group, groups.y, suffix_length);
      } else if (column - conv_dim - inner_dim < num_heads) {
        let local = column - conv_dim - inner_dim;
        split_rows<T, false>(input, dt_out, 0.0, total_dim, column, num_heads, local, row_group, groups.y, suffix_length);
      }
    }
    if (column_groups - group <= groups.x) {
      break;
    }
    group += groups.x;
  }
}
