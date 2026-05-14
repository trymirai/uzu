#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/tensor_view.h"

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(QkUnpack)(
    device const T* qkv,                // [suffix_len, (num_heads + 2*num_groups) * head_dim]
    device T* unpacked_queries,         // [num_heads,   suffix_len,  head_dim]
    device T* unpacked_keys,            // [num_groups,  suffix_len,  head_dim]
    constant uint& head_dim,
    constant uint& num_heads,
    constant uint& num_groups,
    constant uint& suffix_length,
    const uint head_index AXIS(num_heads, 1),
    const uint token_index AXIS(suffix_length, 1),
    const uint dimension_index AXIS(head_dim, 128)
) {
  const uint group_index = head_index / (num_heads / num_groups);
  const uint total_heads = num_heads + 2 * num_groups;
  const uint first_head_in_group = group_index * (num_heads / num_groups);

  TensorView3D<const T> qkv_tensor_view =
      TensorView3D<const T>(qkv).shaped(suffix_length, total_heads, head_dim);
  TensorView3D<T> unpacked_queries_tensor_view =
      TensorView3D<T>(unpacked_queries)
          .shaped(num_heads, suffix_length, head_dim);
  TensorView3D<T> unpacked_keys_tensor_view =
      TensorView3D<T>(unpacked_keys)
          .shaped(num_groups, suffix_length, head_dim);

  unpacked_queries_tensor_view(head_index, token_index, dimension_index) =
      qkv_tensor_view(token_index, head_index, dimension_index);

  if (head_index == first_head_in_group) {
    const uint key_head_index = num_heads + group_index;
    unpacked_keys_tensor_view(group_index, token_index, dimension_index) =
        qkv_tensor_view(token_index, key_head_index, dimension_index);
  }
}
