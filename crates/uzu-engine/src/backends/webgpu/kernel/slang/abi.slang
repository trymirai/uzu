// Storage formats, offsets and physical aliases are backend ABI details.
// The Rust adapter supplies these typed functions when linking compiled WGSL.
#define UZU_STRINGIFY_(X) #X
#define UZU_STRINGIFY(X) UZU_STRINGIFY_(X)
#define UZU_PARAM(T, N) __target_intrinsic(wgsl, UZU_STRINGIFY(uzu_param_##N)) T param_##N();
#define UZU_ARRAY(T, N) __target_intrinsic(wgsl, UZU_STRINGIFY(uzu_array_##N)) T array_##N(uint index);
#define UZU_BUFFER(N)                                                                                        \
    __target_intrinsic(wgsl, UZU_STRINGIFY(load_##N)) float load_##N(uint index);                            \
    __target_intrinsic(wgsl, UZU_STRINGIFY(store_##N)) void store_##N(uint index, float value);              \
    __target_intrinsic(wgsl, UZU_STRINGIFY(word_##N)) uint word_##N(uint index);                             \
    __target_intrinsic(wgsl, UZU_STRINGIFY(put_##N)) void put_##N(uint index, uint value);                   \
    __target_intrinsic(wgsl, UZU_STRINGIFY(round_##N)) float round_##N(float value);

T choose<T>(T a, T b, bool condition) { return condition ? b : a; }
float activate(float v, uint kind, float alpha) {
    switch (kind) {
    case 0:
        return v / (1.0 + exp(-alpha * v));
    case 1:
        return 0.5 * v * (1.0 + tanh(0.7978846 * (v + 0.044715 * v * v * v)));
    case 2: {
        float x = v * 0.7071067811865476;
        float a = abs(x);
        float t = 1.0 / (1.0 + 0.3275911 * a);
        float e = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t +
                         0.254829592) *
                            t * exp(-a * a);
        return 0.5 * v * (1.0 + sign(x) * e);
    }
    case 4:
        return max(v, 0.0) + log(1.0 + exp(-abs(v)));
    default:
        return v;
    }
}
