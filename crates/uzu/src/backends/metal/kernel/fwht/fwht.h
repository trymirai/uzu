#ifndef fwht_h
#define fwht_h

#include <metal_stdlib>

using namespace metal;

#ifndef STEEL_PRAGMA_UNROLL
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#endif

template <short RADIX_SIZE, typename C = float>
METAL_FUNC void radix_func(thread C* elements) {
    constexpr short log_radix = __builtin_ctz(RADIX_SIZE);
    short butterfly_stride = 1;
    STEEL_PRAGMA_UNROLL
    for (short stage = 0; stage < log_radix; stage++) {
        STEEL_PRAGMA_UNROLL
        for (short pair_index = 0; pair_index < RADIX_SIZE / 2; pair_index++) {
            short low_bits = pair_index & (butterfly_stride - 1);
            short base_index = ((pair_index - low_bits) << 1) + low_bits;
            C first = elements[base_index];
            C second = elements[base_index + butterfly_stride];
            elements[base_index] = first + second;
            elements[base_index + butterfly_stride] = first - second;
        }
        butterfly_stride *= 2;
    }
}

#endif
