#pragma once

#include "../../activation/activations.h"
#include "../../common/defines.h"
#include <metal_stdlib>

using namespace metal;

METAL_FUNC float gdn_sigmoid(float x) { return 1.0f / (1.0f + fast::exp(-x)); }

METAL_FUNC float gdn_log_decay(float a_raw, float a_log, float dt_bias) {
  return -fast::exp(a_log) * activate_softplus(a_raw + dt_bias);
}

METAL_FUNC float gdn_decay(float a_raw, float a_log, float dt_bias) {
  return fast::exp(gdn_log_decay(a_raw, a_log, dt_bias));
}

METAL_FUNC float gdn_prefix_decay(float g_row, float g_col) { return fast::exp(g_row - g_col); }
