implementing definitions;

// BF16 in uint16 storage, converted natively in the NATIVE_BF16 modules (VK_KHR_shader_bfloat16 devices), otherwise
// widened exactly and rounded to nearest even like half::bf16::from_f32, which quiets NaNs.
public struct bf16 : IModelFloat16 {
    uint16_t bits;

#ifdef NATIVE_BF16
    public float toFloat() {
        return float(bit_cast<BFloat16>(bits));
    }

    public static bf16 fromFloat(float value) {
        return fromStorageBits(bit_cast<uint16_t>(BFloat16(value)));
    }
#else
    public float toFloat() {
        return asfloat(uint(bits) << 16);
    }

    public static bf16 fromFloat(float value) {
        uint x = asuint(value);
        uint round_bit = 0x8000u;
        if ((x & 0x7FFFFFFFu) > 0x7F800000u) {
            return fromStorageBits((x >> 16) | 0x40u);
        } else if ((x & round_bit) != 0 && (x & (3 * round_bit - 1)) != 0) {
            return fromStorageBits((x >> 16) + 1);
        }
        return fromStorageBits(x >> 16);
    }
#endif

    public uint storageBits() {
        return uint(bits);
    }

    public static bf16 fromStorageBits(uint bits) {
        bf16 result;
        result.bits = uint16_t(bits);
        return result;
    }
}
