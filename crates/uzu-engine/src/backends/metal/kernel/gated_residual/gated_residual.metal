#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/sigmoid_gate.h"

using namespace metal;

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(GatedResidualExpand)(
    const device T* input,
    device T* output,
    constant uint& model_dim,
    constant uint& stream_count,
    constant uint& length,
    const uint position AXIS(length, 256)
) {
  const uint row = position / (stream_count * model_dim);
  const uint channel = position % model_dim;
  output[position] = input[row * model_dim + channel];
}

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(GatedResidualLowRankActivation)(
    device T* values,
    constant uint& stream_count,
    constant uint& length,
    const uint position AXIS(length, 256)
) {
  const float x = float(values[position]) / float(stream_count);
  values[position] = T(uzu::apply_sigmoid_gate(x, x));
}

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(GatedResidualMix)(
    const device T* streams,
    const device T* logits,
    device T* output,
    constant uint& model_dim,
    constant uint& stream_count,
    constant uint& length,
    const uint position AXIS(length, 256)
) {
  const uint row = position / model_dim;
  const uint channel = position % model_dim;
  const uint row_offset = row * stream_count * model_dim + channel;
  float sum = 0.0f;
  for (uint stream = 0; stream < stream_count; ++stream) {
    const uint index = row_offset + stream * model_dim;
    sum += uzu::apply_sigmoid_gate(float(streams[index]), float(logits[index]));
  }
  output[position] = T(sum / float(stream_count));
}

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(GatedResidualUpdate)(
    const device T* streams,
    const device T* injection_logits,
    const device T* update,
    device T* output,
    constant uint& model_dim,
    constant uint& stream_count,
    constant uint& length,
    const uint position AXIS(length, 256)
) {
  const uint row_stream = position / model_dim;
  const uint row = row_stream / stream_count;
  const uint channel = position % model_dim;
  const float injection_logit = float(injection_logits[row_stream]) / float(stream_count);
  const float weight = 2.0f / (1.0f + exp(-injection_logit));
  output[position] = T(float(streams[position]) + weight * float(update[row * model_dim + channel]));
}
