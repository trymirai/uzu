// Included inside MxuFragmentOps; not a standalone header.

template <typename CooperativeTensor, typename U>
METAL_FUNC static void load_paired_vectors(
    thread CooperativeTensor& cooperative,
    const thread ThreadVector<U>& vector_0,
    const thread ThreadVector<U>& vector_1
) {
  if constexpr (RELAXED) {
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
      cooperative[i] = vector_0[i];
      cooperative[ELEMENTS_PER_THREAD + i] = vector_1[i];
    }
  } else {
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < 4; i++) {
      cooperative[i] = vector_0[i];
      cooperative[4 + i] = vector_1[i];
      cooperative[8 + i] = vector_0[4 + i];
      cooperative[12 + i] = vector_1[4 + i];
    }
  }
}

template <typename CooperativeTensor, typename U>
METAL_FUNC static void store_paired_vectors(
    thread CooperativeTensor& cooperative,
    thread ThreadVector<U>& vector_0,
    thread ThreadVector<U>& vector_1
) {
  if constexpr (RELAXED) {
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
      vector_0[i] = cooperative[i];
      vector_1[i] = cooperative[ELEMENTS_PER_THREAD + i];
    }
  } else {
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < 4; i++) {
      vector_0[i] = cooperative[i];
      vector_1[i] = cooperative[4 + i];
      vector_0[4 + i] = cooperative[8 + i];
      vector_1[4 + i] = cooperative[12 + i];
    }
  }
}
