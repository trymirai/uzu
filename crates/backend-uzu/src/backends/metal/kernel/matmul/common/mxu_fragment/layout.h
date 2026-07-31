UZU_CONST ushort FRAGMENT_ROWS = MXU_MMA_ROWS;
UZU_CONST ushort FRAGMENT_COLS = MXU_MMA_COLS;
UZU_CONST bool READ_TRANSPOSE_SWAPS_SOURCE_STRIDES = false;
using BlockStorage = DeviceBlockStorage;

UZU_CONST ushort ELEMENTS_PER_THREAD = (FRAGMENT_ROWS * FRAGMENT_COLS) / METAL_SIMD_SIZE;

UZU_CONST ushort THREAD_ELEMENT_ROWS = 2;
UZU_CONST ushort THREAD_ELEMENT_COLS = 4;

UZU_CONST ushort THREAD_ELEMENT_ROW_STRIDE = FRAGMENT_ROWS / THREAD_ELEMENT_ROWS;

static_assert(
    THREAD_ELEMENT_ROWS * THREAD_ELEMENT_COLS == ELEMENTS_PER_THREAD,
    "MxuFragment shape is not consistent with element count"
);

template <typename U>
using ThreadVector = typename metal::vec<U, ELEMENTS_PER_THREAD>;

METAL_FUNC static constexpr short2 get_position(ushort simd_lane_id) {
  if constexpr (RELAXED) {
    const short quad = simd_lane_id / 4;
    const short row = (quad & 4) + (simd_lane_id / 2) % 4;
    const short col = ((quad & 2) + simd_lane_id % 2) * THREAD_ELEMENT_COLS;
    return short2{col, row};
  } else {
    const short col = short((simd_lane_id & 1) * 2 + ((simd_lane_id >> 3) & 1) * 4);
    const short row = short(((simd_lane_id >> 1) & 3) + ((simd_lane_id >> 4) & 1) * 4);
    return short2{col, row};
  }
}

METAL_FUNC static constexpr short2 get_element_offset(ushort element_index) {
  if constexpr (RELAXED) {
    const short row = short((element_index / THREAD_ELEMENT_COLS) * THREAD_ELEMENT_ROW_STRIDE);
    const short col = short(element_index % THREAD_ELEMENT_COLS);
    return short2{col, row};
  } else {
    const short row = short((element_index / 4) * 8);
    const ushort col_slot = element_index & 3;
    const short col = short((col_slot & 1) + (col_slot / 2) * 8);
    return short2{col, row};
  }
}
