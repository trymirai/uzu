#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

#include "../../matmul/common/defines.h"

namespace uzu {
namespace quantized_matmul {

template <int bits, int word_size = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : word_size / bits);
}

template <int bits, int word_size = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (word_size / 8) : (bits == 5 ? 5 : 3);
}

template <typename T, typename AccumulatorType, int values_per_thread, int bits>
inline AccumulatorType load_input_vector(
    const device T* input,
    thread AccumulatorType* input_thread
) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  AccumulatorType sum = 0;
  if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += input[i] + input[i + 1] + input[i + 2] + input[i + 3];
      input_thread[i] = input[i];
      input_thread[i + 1] = input[i + 1] / 16.0f;
      input_thread[i + 2] = input[i + 2] / 256.0f;
      input_thread[i + 3] = input[i + 3] / 4096.0f;
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += input[i];
      input_thread[i] = input[i];
    }
  }
  return sum;
}

template <typename T, typename AccumulatorType, int values_per_thread, int bits>
inline AccumulatorType load_input_vector_checked(
    const device T* input,
    thread AccumulatorType* input_thread,
    int valid_count
) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  AccumulatorType sum = 0;
  if (bits == 4) {
    const AccumulatorType scale_lookup[4] = {
        static_cast<AccumulatorType>(1.0f),
        static_cast<AccumulatorType>(1.0f / 16.0f),
        static_cast<AccumulatorType>(1.0f / 256.0f),
        static_cast<AccumulatorType>(1.0f / 4096.0f)
    };

    for (int i = 0; i < values_per_thread; ++i) {
      input_thread[i] = 0;
    }
    for (int i = 0; i < valid_count; ++i) {
      AccumulatorType value = input[i];
      sum += value;
      input_thread[i] = value * scale_lookup[i & 3];
    }
  } else if (bits == 8) {
    for (int i = 0; i < valid_count; ++i) {
      AccumulatorType value = input[i];
      sum += value;
      input_thread[i] = value;
    }
    for (int i = valid_count; i < values_per_thread; ++i) {
      input_thread[i] = 0;
    }
  }
  return sum;
}

template <typename AccumulatorType, int values_per_thread, int bits>
inline void quantized_outer_product(
    const thread uint8_t* weights,
    AccumulatorType input_value,
    AccumulatorType scale,
    AccumulatorType bias,
    thread AccumulatorType* result
) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  if (bits == 4) {
    AccumulatorType scale_low = scale;
    AccumulatorType scale_high = scale / 16.0f;
    for (int i = 0; i < (values_per_thread / 2); i++) {
      result[2 * i] += input_value * (scale_low * (weights[i] & 0x0f) + bias);
      result[2 * i + 1] +=
          input_value * (scale_high * (weights[i] & 0xf0) + bias);
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      result[i] += input_value * (scale * weights[i] + bias);
    }
  }
}

template <typename AccumulatorType, int values_per_thread, int bits>
inline AccumulatorType quantized_dot_product(
    const device uint8_t* weights,
    const thread AccumulatorType* input_thread,
    AccumulatorType scale,
    AccumulatorType bias,
    AccumulatorType input_sum
) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  AccumulatorType accumulator = 0;
  if (bits == 4) {
    const device uint16_t* weight_pairs = (const device uint16_t*)weights;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accumulator +=
          (input_thread[4 * i] * (weight_pairs[i] & 0x000f) +
           input_thread[4 * i + 1] * (weight_pairs[i] & 0x00f0) +
           input_thread[4 * i + 2] * (weight_pairs[i] & 0x0f00) +
           input_thread[4 * i + 3] * (weight_pairs[i] & 0xf000));
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accumulator += input_thread[i] * weights[i];
    }
  }
  return scale * accumulator + input_sum * bias;
}

template <typename AccumulatorType, int values_per_thread, int bits>
inline AccumulatorType quantized_dot_product_zero_point(
    const device uint8_t* weights,
    const thread AccumulatorType* input_thread,
    AccumulatorType scale,
    AccumulatorType zero_point
) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  AccumulatorType accumulator = 0;
  if (bits == 4) {
    const device uint16_t* weight_pairs = (const device uint16_t*)weights;
    const uint16_t zero_point_0 = static_cast<uint16_t>(zero_point);
    const uint16_t zero_point_1 = static_cast<uint16_t>(zero_point) << 4;
    const uint16_t zero_point_2 = static_cast<uint16_t>(zero_point) << 8;
    const uint16_t zero_point_3 = static_cast<uint16_t>(zero_point) << 12;

    for (int i = 0; i < (values_per_thread / 4); i++) {
      uint16_t word = weight_pairs[i];
      accumulator += input_thread[4 * i] *
                     static_cast<AccumulatorType>(
                         static_cast<int>(word & 0x000f) -
                         static_cast<int>(zero_point_0)
                     );
      accumulator += input_thread[4 * i + 1] *
                     static_cast<AccumulatorType>(
                         static_cast<int>(word & 0x00f0) -
                         static_cast<int>(zero_point_1)
                     );
      accumulator += input_thread[4 * i + 2] *
                     static_cast<AccumulatorType>(
                         static_cast<int>(word & 0x0f00) -
                         static_cast<int>(zero_point_2)
                     );
      accumulator += input_thread[4 * i + 3] *
                     static_cast<AccumulatorType>(
                         static_cast<int>(word & 0xf000) -
                         static_cast<int>(zero_point_3)
                     );
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accumulator +=
          input_thread[i] * (static_cast<AccumulatorType>(weights[i]) - zero_point);
    }
  }
  return scale * accumulator;
}

template <typename AccumulatorType, int values_per_thread, int bits>
inline AccumulatorType quantized_dot_product_checked(
    const device uint8_t* weights,
    const thread AccumulatorType* input_thread,
    AccumulatorType scale,
    AccumulatorType bias,
    AccumulatorType input_sum,
    int valid_count
) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  AccumulatorType accumulator = 0;
  if (bits == 4) {
    const device uint16_t* weight_pairs = (const device uint16_t*)weights;

    int full_pairs = valid_count / 4;
    for (int i = 0; i < full_pairs; i++) {
      accumulator +=
          (input_thread[4 * i] * (weight_pairs[i] & 0x000f) +
           input_thread[4 * i + 1] * (weight_pairs[i] & 0x00f0) +
           input_thread[4 * i + 2] * (weight_pairs[i] & 0x0f00) +
           input_thread[4 * i + 3] * (weight_pairs[i] & 0xf000));
    }

    int remainder = valid_count & 3;
    if (remainder > 0) {
      uint16_t word_value = weight_pairs[full_pairs];
      int base = 4 * full_pairs;
      if (remainder > 0)
        accumulator += input_thread[base] * (word_value & 0x000f);
      if (remainder > 1)
        accumulator += input_thread[base + 1] * (word_value & 0x00f0);
      if (remainder > 2)
        accumulator += input_thread[base + 2] * (word_value & 0x0f00);
    }
  } else if (bits == 8) {
    for (int i = 0; i < valid_count; i++) {
      accumulator += input_thread[i] * weights[i];
    }
  }

  return scale * accumulator + input_sum * bias;
}

template <typename AccumulatorType, int element_count, int bits>
inline void dequantize(
    const device uint8_t* weights,
    AccumulatorType scale,
    AccumulatorType bias,
    threadgroup AccumulatorType* destination
) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  if (bits == 4) {
    AccumulatorType scale_low = scale;
    AccumulatorType scale_high = scale / static_cast<AccumulatorType>(16.0f);
    for (int i = 0; i < (element_count / 2); i++) {
      destination[2 * i] = scale_low * (weights[i] & 0x0f) + bias;
      destination[2 * i + 1] = scale_high * (weights[i] & 0xf0) + bias;
    }
  } else if (bits == 8) {
    for (int i = 0; i < element_count; i++) {
      destination[i] = scale * weights[i] + bias;
    }
  }
}

template <>
inline void dequantize<bfloat, 8, 4>(
    const device uint8_t* weights,
    bfloat scale,
    bfloat bias,
    threadgroup bfloat* destination
) {
  const device uint32_t* weight_pointer = (const device uint32_t*)weights;
  uint32_t packed = *weight_pointer;

  bfloat4 low_nibbles, high_nibbles;

  low_nibbles.x = static_cast<bfloat>(packed & 0xF);
  low_nibbles.y = static_cast<bfloat>((packed >> 4) & 0xF);
  low_nibbles.z = static_cast<bfloat>((packed >> 8) & 0xF);
  low_nibbles.w = static_cast<bfloat>((packed >> 12) & 0xF);

  high_nibbles.x = static_cast<bfloat>((packed >> 16) & 0xF);
  high_nibbles.y = static_cast<bfloat>((packed >> 20) & 0xF);
  high_nibbles.z = static_cast<bfloat>((packed >> 24) & 0xF);
  high_nibbles.w = static_cast<bfloat>((packed >> 28) & 0xF);

  low_nibbles = low_nibbles * scale + bias;
  high_nibbles = high_nibbles * scale + bias;

  threadgroup bfloat4* output_pointer = (threadgroup bfloat4*)destination;
  output_pointer[0] = low_nibbles;
  output_pointer[1] = high_nibbles;
}

} // namespace quantized_matmul
} // namespace uzu
