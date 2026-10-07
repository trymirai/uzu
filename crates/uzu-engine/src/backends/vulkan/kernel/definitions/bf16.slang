implementing definitions;

// BF16 in uint16 storage, needing no BF16 device extension. Loads widen to the upper half of an FP32 and
// stores round to nearest even, both exactly like half::bf16 (to_f32 / from_f32), quieting NaNs.
public struct bf16 : IStorageFloat {
    public uint16_t bits;

    public float toFloat() {
        uint x = uint(bits);
        if ((x & 0x7FFFu) > 0x7F80u) { x |= 0x40u; }
        return asfloat(x << 16);
    }

    public static bf16 fromFloat(float value) {
        uint x = asuint(value);
        uint round_bit = 0x8000u;
        bf16 result;
        if ((x & 0x7FFFFFFFu) > 0x7F800000u) {
            result.bits = uint16_t((x >> 16) | 0x40u);
        } else if ((x & round_bit) != 0 && (x & (3 * round_bit - 1)) != 0) {
            result.bits = uint16_t((x >> 16) + 1);
        } else {
            result.bits = uint16_t(x >> 16);
        }
        return result;
    }
}
