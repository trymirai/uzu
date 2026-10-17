implementing definitions;

// Element type of a device buffer. Kernels compute in FP32 and convert only at loads and stores.
public interface IStorageFloat {
    public float toFloat();
    public static This fromFloat(float value);
    // The stored bits, zero-extended, and their inverse: exact, for moving typed values between lanes.
    public uint storageBits();
    public static This fromStorageBits(uint bits);
    // The value of the stored bits in the low bits of `bits`, those above ignored, widened without toFloat's NaN
    // quieting: finite values alike, NaNs and infinities with exponent field 255, for staging that classifies values by
    // their FP32 fields.
    public static float widened(uint bits);
    // Explicitly stored fraction bits.
    public static const uint FRACTION_BITS;
}

public extension float : IStorageFloat {
    public float toFloat() {
        return this;
    }

    public static float fromFloat(float value) {
        return value;
    }

    public uint storageBits() {
        return asuint(this);
    }

    public static float fromStorageBits(uint bits) {
        return asfloat(bits);
    }

    public static float widened(uint bits) {
        return asfloat(bits);
    }

    public static const uint FRACTION_BITS = 23;
}

// Narrowing rounds to nearest even in integer arithmetic on the FP32 bits. Drivers compile a native conversion widened
// again without its intermediate F16 rounding: lavapipe and Mesa without NoContraction on the conversion, NVIDIA even
// with it and through any widening, including the core module's float(half) that concrete half.toFloat() calls reach.
// Narrowed bits leave no conversion to fold. Widening goes through the half's bits, as for those drivers.
public extension half : IStorageFloat {
    public float toFloat() {
        return widened(storageBits());
    }

    public static float widened(uint bits) {
        return spirv_asm {
            %pair:$$float2 = OpExtInst glsl450 UnpackHalf2x16 $bits;
            result:$$float = OpCompositeExtract %pair 0;
        };
    }

    public static half fromFloat(float value) {
        let bits = asuint(value);
        let sign = (bits >> 16) & 0x8000u;
        let magnitude = bits & 0x7fffffffu;
        if (magnitude > 0x7f800000u) {
            // NaN, quieted, the payload's high bits kept.
            return asfloat16(uint16_t(sign | 0x7e00u | ((magnitude >> 13) & 0x3ffu)));
        }
        if (magnitude >= 0x477ff000u) {
            // At least 65520, halfway past the largest half: infinity.
            return asfloat16(uint16_t(sign | 0x7c00u));
        }
        // Normal halves keep the exponent rebiased by 15 - 127 and drop 13 fraction bits; subnormal ones drop
        // 126 - exponent bits of the significand, whose units are then 2^-24 (a carry into the exponent field is the
        // correctly rounded next binade). Below 2^-25 the value rounds to zero.
        let exponent = magnitude >> 23;
        if (exponent < 102u) {
            return asfloat16(uint16_t(sign));
        }
        let normal = exponent >= 113u;
        let significand = normal ? magnitude & 0x7fffffu : (magnitude & 0x7fffffu) | 0x800000u;
        let shift = normal ? 13u : 126u - exponent;
        let half_unit = 1u << (shift - 1u);
        let dropped = significand & ((1u << shift) - 1u);
        var result = (normal ? (exponent - 112u) << 10 : 0u) | (significand >> shift);
        if (dropped > half_unit || (dropped == half_unit && (result & 1u) != 0u)) {
            result += 1u;
        }
        return asfloat16(uint16_t(sign | result));
    }

    public uint storageBits() {
        return uint(asuint16(this));
    }

    public static half fromStorageBits(uint bits) {
        return asfloat16(uint16_t(bits));
    }

    public static const uint FRACTION_BITS = 10;
}
