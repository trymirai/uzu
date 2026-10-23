import definitions;

public static const uint GEMV_MAX_SLICE_WORDS = 2;

public struct QuantChunk {
  public static const uint VALUES = 8;
  public static const uint FLOAT_MANTISSA_BITS = 23;
  public static const uint FLOAT_EXPONENT_BIAS = 127;
  public static const uint MANTISSA_BASE_BITS = (FLOAT_EXPONENT_BIAS + FLOAT_MANTISSA_BITS) << FLOAT_MANTISSA_BITS;
  public static const float MANTISSA_BASE = 8388608.0;
  public static const uint BITS_PER_BYTE = 8;
  public static const uint WORD_BITS = 32;
  public static const uint WORD_LANES = 4;
  public static const uint WORD_BYTES = WORD_BITS / BITS_PER_BYTE;
  public static const uint VECTOR_BYTES = WORD_BYTES * WORD_LANES;
  public static const uint STORAGE_BITS = 16;
  public static const uint VALUES_PER_WORD = WORD_BITS / STORAGE_BITS;

  public static void decode(
    uint4 words[GEMV_MAX_SLICE_WORDS],
    uint chunk,
    out float values[VALUES],
    float origin,
    bool signed_codes,
    uint BITS
  ) {
    let word = chunk * (BITS == 4 ? 1u : 2u);
    let word0 = words[word / 4u][word % 4u];
    let word1 = BITS == 4 ? 0u : words[(word + 1u) / 4u][(word + 1u) % 4u];
    decode_words(word0, word1, values, origin, signed_codes, BITS);
  }

  // Values `offset` to offset + 7 of row `row` of `extent` values of `source`.
  public static void load<A : IModelFloat>(
    bool ALIGNED,
    Ptr<A, Access.Read> source,
    out float values[VALUES],
    uint offset,
    uint extent,
    uint row
  ) {
    let PACKED_16 = __naturalStrideOf<A>() == 2;
    let element = row * extent + offset;
    if (ALIGNED) {
      load_values(false, PACKED_16, source, element, values, VALUES);
    } else {
      let valid = offset < extent ? min(extent - offset, VALUES) : 0u;
      let source_aligned = !PACKED_16 || (element & 7u) == 0;
      if (valid == VALUES && source_aligned) {
        load_values(false, PACKED_16, source, element, values, VALUES);
      } else {
        load_values(true, PACKED_16, source, element, values, valid);
      }
    }
  }

  static void decode_words(uint packed0, uint packed1, out float values[VALUES], float origin, bool signed_codes, uint BITS) {
    let CODES_PER_WORD = WORD_BITS / BITS;
    let W4_SIGN_MASK = 0x88888888u;
    let W8_SIGN_BITS = 0x80808080u;
    let bits_mask = (1u << BITS) - 1u;
    var word0 = packed0;
    var word1 = packed1;
    if (BITS == 4) {
      word0 ^= signed_codes ? W4_SIGN_MASK : 0u;
      [unroll]
      for (uint i = 0; i < VALUES; i++) {
        values[i] = asfloat(((word0 >> (BITS * i)) & bits_mask) | MANTISSA_BASE_BITS) - origin;
      }
    } else {
      let mask = signed_codes ? W8_SIGN_BITS : 0u;
      word0 ^= mask;
      word1 ^= mask;
      [unroll]
      for (uint i = 0; i < CODES_PER_WORD; i++) {
        values[i] = asfloat(((word0 >> (BITS * i)) & bits_mask) | MANTISSA_BASE_BITS) - origin;
        values[CODES_PER_WORD + i] = asfloat(((word1 >> (BITS * i)) & bits_mask) | MANTISSA_BASE_BITS) - origin;
      }
    }
  }

  static void load_values<A : IModelFloat>(
    bool BOUNDED,
    bool PACKED_16,
    Ptr<A, Access.Read> source,
    uint element,
    out float values[VALUES],
    uint valid
  ) {
    if (!BOUNDED && PACKED_16) {
      let packed = load_aligned<uint4, 16>(source, element);
      [unroll]
      for (uint i = 0; i < WORD_LANES; i++) {
        values[VALUES_PER_WORD * i] = A.fromStorageBits(packed[i]).toFloat();
        values[VALUES_PER_WORD * i + 1] = A.fromStorageBits(packed[i] >> STORAGE_BITS).toFloat();
      }
    } else {
      [unroll]
      for (uint i = 0; i < VALUES; i++) {
        var value = 0.0;
        if (!BOUNDED || i < valid) {
          value = source[element + i].toFloat();
        }
        values[i] = value;
      }
    }
  }
}
