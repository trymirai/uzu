#pragma once

#include "../../common/defines.h"
#include "../../matmul/common/fragment.h"
#include "../../matmul/common/mxu_fragment_ops.h"
#include "../../matmul/common/simdgroup_fragment_ops.h"
#include <metal_stdlib>

using namespace metal;
using namespace uzu::matmul;

template <typename AccFragment, typename LeftFragment, typename RightFragment, typename T>
METAL_FUNC void gdn_accumulate_dual_gram_tile(
    thread AccFragment& kk_acc,
    thread AccFragment& qk_acc,
    const device T* k_rows,
    const device T* q_rows,
    const device T* k_cols,
    const int stride,
    const ushort valid_rows,
    const ushort valid_cols,
    const ushort valid_k,
    const bool full_left,
    const bool full_right,
    const ushort lane
) {
  LeftFragment k_left;
  LeftFragment q_left;
  RightFragment k_right;

  if (full_left) {
    k_left.load_from(lane, fragment_source(k_rows, stride));
    q_left.load_from(lane, fragment_source(q_rows, stride));
  } else {
    k_left.load_from(lane, fragment_source(k_rows, stride).bounded(valid_rows, valid_k));
    q_left.load_from(lane, fragment_source(q_rows, stride).bounded(valid_rows, valid_k));
  }

  if (full_right) {
    k_right.load_from(lane, fragment_source(k_cols, stride));
  } else {
    k_right.load_from(lane, fragment_source(k_cols, stride).bounded(valid_cols, valid_k));
  }

  fragment_mma(kk_acc, k_left, k_right);
  fragment_mma(qk_acc, q_left, k_right);
}
