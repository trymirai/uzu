#include <metal_stdlib>
#include "../common/dsl.h"

template <typename InputT, typename OutputT>
VARIANTS(InputT, bfloat)
VARIANTS(OutputT, float)
PUBLIC KERNEL(TensorCast)(
    const device InputT* input,
    device OutputT* output,
    constant uint& length,
    const uint position AXIS(length, 256)
) {
    output[position] = static_cast<OutputT>(input[position]);
}
