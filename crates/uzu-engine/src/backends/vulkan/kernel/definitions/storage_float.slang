implementing definitions;

public interface IStorageFloat {
    public float toFloat();
    public static This fromFloat(float value);
    public uint storageBits();
    public static This fromStorageBits(uint bits);
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
}
