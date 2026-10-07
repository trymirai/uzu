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

// Narrowing rounds to nearest even under the RoundingModeRTE 16 execution mode every generated entry point declares.
// Widening goes through the half's bits: compilers may drop a plain float(half(x)) round trip, losing the rounding.
public extension half : IStorageFloat {
    public float toFloat() {
        let bits = uint(asuint16(this));
        return spirv_asm {
            %pair:$$float2 = OpExtInst glsl450 UnpackHalf2x16 $bits;
            result:$$float = OpCompositeExtract %pair 0;
        };
    }

    public static half fromFloat(float value) {
        return half(value);
    }
}
