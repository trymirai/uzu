import generated.kv_layout;

public float attentionSign(uint32_t channel) {
  return ((ROTATION_SIGN_BITS[channel / WORD_BITS % 8] >> (channel % WORD_BITS)) & 1) != 0 ? -1.0 : 1.0;
}

// The portable layout uses the same normalized rotation as the CPU backend.
public void attentionHadamard<let N : uint32_t>(inout float values[N]) {
  for (uint32_t stride = 1; stride < N; stride *= 2) {
    for (uint32_t channel = 0; channel < N; ++channel) {
      if ((channel & stride) == 0) {
        float a = values[channel];
        float b = values[channel | stride];
        values[channel] = a + b;
        values[channel | stride] = a - b;
      }
    }
  }
  float scale = 1.0 / sqrt(float(N));
  for (uint32_t channel = 0; channel < N; ++channel)
    values[channel] *= scale;
}

public int32_t attentionQuantize(float value, float scale) {
  float magnitude = abs(value);
  int32_t lower = min(int32_t(floor(magnitude / scale)), 127);
  // Compare in the original units: reciprocal-based division can perturb an
  // exact half-integer. A BF16 scale times this half-integer is exact in f32.
  float midpoint = (float(lower) + 0.5) * scale;
  int32_t code = lower + ((magnitude > midpoint || (magnitude == midpoint && (lower & 1) != 0)) ? 1 : 0);
  return min(code, 127) * (value < 0.0 ? -1 : 1);
}

public int32_t attentionSignedCode(uint32_t code) {
  return int32_t(code ^ 128) - 128;
}
