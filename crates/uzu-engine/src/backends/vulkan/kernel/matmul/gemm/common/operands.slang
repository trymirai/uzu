import definitions;
import matmul.common.matmul;
import generated.gemm;
import matmul.common.loader;
import matmul.common.simdgroup_multiply_accumulate;

// The right operand B as stored, N rows of K, full precision or U4/U8 codes, and the loader that stages a
// block_n x block_k tile of it, rows shared_stride_b apart.
public interface IRightOperand<B> {
  associatedtype Loader : IThreadgroupLoader<B>;

  public Loader make_loader(
    uint block_col,
    int block_n,
    int block_k,
    int shared_stride_b,
    int threadgroup_threads,
    ThreadContext thread_context,
  );
}

// Full-precision B.
public struct FullPrecisionRight<B : IStorageFloat> : IRightOperand<B> {
  public DenseOperand<B> dense;

  public typealias Loader = ThreadgroupLoader<B>;

  public Loader make_loader(
    uint block_col,
    int block_n,
    int block_k,
    int shared_stride_b,
    int threadgroup_threads,
    ThreadContext thread_context,
  ) {
    return ThreadgroupLoader<B>(
      dense.values,
      block_col * dense.k,
      int(dense.k),
      block_n,
      block_k,
      shared_stride_b,
      threadgroup_threads,
      thread_context
    );
  }
}

// U4/U8 B of K-element rows under any scheme.
public struct QuantizedRight<B : IStorageFloat> : IRightOperand<B> {
  public QuantizedWeights<B> weights;

  public typealias Loader = QuantizedBlockLoader<B>;

  public Loader make_loader(
    uint block_col,
    int block_n,
    int block_k,
    int shared_stride_b,
    int threadgroup_threads,
    ThreadContext thread_context,
  ) {
    return QuantizedBlockLoader<B>(
      weights, block_col, block_n, block_k, shared_stride_b, threadgroup_threads, thread_context
    );
  }
}

// Stages a block_n x block_k tile of quantized B, each invocation dequantizing reads_per_thread consecutive packs of one
// row with its group's scale and staged bias in T arithmetic.
public struct QuantizedBlockLoader<T : IStorageFloat> : IThreadgroupLoader<T> {
  QuantizedWeights<T> weights;
  int threadgroup_tile_rows;
  int pack_factor;
  int reads_per_thread;
  int group_steps_per_block;
  bool tile_has_idle_threads;

  int tile_stride;
  int group_step_counter;

  int tile_row_index;
  int tile_col_index;

  uint dst;
  uint src_index;
  uint row;
  uint group;

  public __init(
    QuantizedWeights<T> right,
    uint block_col,
    int tile_rows,
    int tile_cols,
    int destination_leading_dimension,
    int threadgroup_size,
    ThreadContext thread_context,
  ) {
    weights = right;
    threadgroup_tile_rows = tile_rows;
    pack_factor = right.u4 ? 2 : 1;
    let tile_cols_packed = tile_cols / pack_factor;
    tile_has_idle_threads = tile_cols_packed * tile_rows < threadgroup_size;
    reads_per_thread = tile_has_idle_threads ? 1 : (tile_cols_packed * tile_rows) / threadgroup_size;
    group_steps_per_block = int(right.group_size) / tile_cols;

    tile_stride = tile_cols_packed;
    group_step_counter = 0;
    let thread_index = int(thread_context.thread_index);
    tile_row_index = reads_per_thread * thread_index / tile_cols_packed;
    tile_col_index = (reads_per_thread * thread_index) % tile_cols_packed;
    dst = uint(tile_row_index * destination_leading_dimension + tile_col_index * pack_factor);
    row = block_col + uint(tile_row_index);
    src_index = right.code_byte(row, 0) + uint(tile_col_index);
    group = 0;
  }

  // Code byte i of this invocation's run into S.
  void stage<S : IThreadgroupStorage<T>>(int i, float scale, float bias) {
    T values[2];
    dequantize_code<T>(uint(weights.codes[src_index + uint(i)]), scale, bias, weights.u4, weights.signed_codes, values);
    S.write(dst + uint(i * pack_factor), values[0]);
    if (weights.u4) {
      S.write(dst + uint(i * pack_factor + 1), values[1]);
    }
  }

  [mutating]
  public void load_unsafe<S : IThreadgroupStorage<T>>() {
    if (tile_has_idle_threads && tile_row_index >= threadgroup_tile_rows) {
      return;
    }
    let metadata = weights.metadata(row, group);
    let bias = staged_bias<T>(metadata, weights.method);
    for (int i = 0; i < reads_per_thread; i++) {
      stage<S>(i, metadata.scale, bias);
    }
  }

  [mutating]
  public void load_safe<S : IThreadgroupStorage<T>>(int2 src_tile_dim) {
    if (tile_has_idle_threads && tile_row_index >= threadgroup_tile_rows) {
      return;
    }
    let zero = T.fromFloat(0.0);
    if (tile_row_index >= src_tile_dim.y) {
      for (int i = 0; i < reads_per_thread * pack_factor; i++) {
        S.write(dst + uint(i), zero);
      }
      return;
    }
    let valid_cols = src_tile_dim.x;
    let valid_packs = (valid_cols + pack_factor - 1) / pack_factor;
    let metadata = weights.metadata(row, group);
    let bias = staged_bias<T>(metadata, weights.method);
    for (int i = 0; i < reads_per_thread; i++) {
      let pack_index = tile_col_index + i;
      if (pack_index < valid_packs) {
        stage<S>(i, metadata.scale, bias);
        if (pack_index == valid_packs - 1) {
          for (int lane = valid_cols - pack_index * pack_factor; lane < pack_factor; ++lane) {
            S.write(dst + uint(i * pack_factor + lane), zero);
          }
        }
      } else {
        for (int lane = 0; lane < pack_factor; ++lane) {
          S.write(dst + uint(i * pack_factor + lane), zero);
        }
      }
    }
  }

  [mutating]
  public void next() {
    src_index += uint(tile_stride);
    if (group_steps_per_block > 1) {
      group_step_counter++;
      if (group_step_counter != group_steps_per_block) {
        return;
      }
      group_step_counter = 0;
    }
    group++;
  }
}
