implementing definitions;

// Element type of a device buffer. Kernels compute in FP32 and convert only at loads and stores.
public interface IStorageFloat {
    public float toFloat();
    public static This fromFloat(float value);
}

public extension float : IStorageFloat {
    public float toFloat() {
        return this;
    }

    public static float fromFloat(float value) {
        return value;
    }
}

public extension half : IStorageFloat {
    public float toFloat() {
        return float(this);
    }

    public static half fromFloat(float value) {
        return half(value);
    }
}
