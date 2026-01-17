#include <metal_stdlib>
using namespace metal;

constexpr sampler bicubicSampler(filter::linear, address::clamp_to_edge);

struct ImageParameters {
  uint2 inputDimensions;
  float3 imageMean;
  float3 imageStd;
  float paddingValueRgb;
};

struct PatchParameters {
  uint2 paddedDimensions;
  uint patchSize;
  uint numChannels;
  uint temporalSlices;
};

[[max_total_threads_per_threadgroup(1024)]]
kernel void scalePadNormalizeImage(
    texture2d<float, access::sample> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant ImageParameters& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {

  uint2 outputDims =
      uint2(outputTexture.get_width(), outputTexture.get_height());

  if (gid.x >= outputDims.x || gid.y >= outputDims.y) {
    return;
  }

  float inputAspectRatio =
      float(params.inputDimensions.x) / float(params.inputDimensions.y);
  float outputAspectRatio = float(outputDims.x) / float(outputDims.y);

  float scaleFactor;
  if (inputAspectRatio > outputAspectRatio) {
    scaleFactor = float(outputDims.x) / float(params.inputDimensions.x);
  } else {
    scaleFactor = float(outputDims.y) / float(params.inputDimensions.y);
  }

  float scaledInputWidth = float(params.inputDimensions.x) * scaleFactor;
  float scaledInputHeight = float(params.inputDimensions.y) * scaleFactor;

  float offsetX = (float(outputDims.x) - scaledInputWidth) / 2.0f;
  float offsetY = (float(outputDims.y) - scaledInputHeight) / 2.0f;

  float targetXInScaledArea = float(gid.x) - offsetX;
  float targetYInScaledArea = float(gid.y) - offsetY;

  float4 pixelColor;

  if (targetXInScaledArea >= 0.0f && targetXInScaledArea < scaledInputWidth &&
      targetYInScaledArea >= 0.0f && targetYInScaledArea < scaledInputHeight) {

    float uNorm = targetXInScaledArea / scaledInputWidth;
    float vNorm = targetYInScaledArea / scaledInputHeight;

    pixelColor = inputTexture.sample(bicubicSampler, float2(uNorm, vNorm));
  } else {
    pixelColor = float4(
        params.paddingValueRgb,
        params.paddingValueRgb,
        params.paddingValueRgb,
        1.0f
    );
  }

  float3 normalizedRgb;
  float epsilon = 1e-6f;
  normalizedRgb.r =
      (pixelColor.r - params.imageMean.r) / (params.imageStd.r + epsilon);
  normalizedRgb.g =
      (pixelColor.g - params.imageMean.g) / (params.imageStd.g + epsilon);
  normalizedRgb.b =
      (pixelColor.b - params.imageMean.b) / (params.imageStd.b + epsilon);

  outputTexture.write(float4(normalizedRgb, 1.0f), gid);
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void extractImagePatches(
    texture2d<float, access::read> paddedNormalizedTexture [[texture(0)]],
    device float* outputBuffer [[buffer(0)]],
    constant PatchParameters& params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {

  if (gid.x >= params.paddedDimensions.x ||
      gid.y >= params.paddedDimensions.y) {
    return;
  }

  uint numPatchesX = params.paddedDimensions.x / params.patchSize;

  uint patchCol = gid.x / params.patchSize;
  uint patchRow = gid.y / params.patchSize;

  uint spatialPatchIdxFlat = patchRow * numPatchesX + patchCol;

  uint pxInPatch = gid.x % params.patchSize;
  uint pyInPatch = gid.y % params.patchSize;

  float4 normalizedPixelRgba = paddedNormalizedTexture.read(gid);

  uint patchPixelFlatIdx = pyInPatch * params.patchSize + pxInPatch;
  uint singlePatchChannelTemporalSliceSize =
      params.patchSize * params.patchSize;
  uint singlePatchChannelSize =
      params.temporalSlices * singlePatchChannelTemporalSliceSize;
  uint singlePatchSize = params.numChannels * singlePatchChannelSize;

  uint outputPatchBaseIdx = spatialPatchIdxFlat * singlePatchSize;

  for (uint c = 0; c < params.numChannels; ++c) {
    float valueToStore = normalizedPixelRgba[c];
    for (uint t = 0; t < params.temporalSlices; ++t) {
      uint channelBlockOffset = c * singlePatchChannelSize;
      uint temporalBlockOffset = t * singlePatchChannelTemporalSliceSize;

      uint outputBufferIdx = outputPatchBaseIdx + channelBlockOffset +
                             temporalBlockOffset + patchPixelFlatIdx;

      outputBuffer[outputBufferIdx] = valueToStore;
    }
  }
}