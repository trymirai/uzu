implementing definitions;

// Element type of a device buffer. Kernels compute in FP32 and convert only at loads and stores.
public interface IStorageFloat {
    public float toFloat();
    public static This fromFloat(float value);
    // The stored bits, zero-extended, and their inverse, which ignores the bits above them: exact, for moving typed
    // values between lanes and for values loaded as packed words.
    public uint storageBits();
    public static This fromStorageBits(uint bits);
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

    public static const uint FRACTION_BITS = 23;
}

public extension half : IStorageFloat {
    public float toFloat() {
        return float(this);
    }

    public static half fromFloat(float value) {
        return half(value);
    }

    public uint storageBits() {
        return uint(asuint16(this));
    }

    public static half fromStorageBits(uint bits) {
        return asfloat16(uint16_t(bits));
    }

    public static const uint FRACTION_BITS = 10;
}
