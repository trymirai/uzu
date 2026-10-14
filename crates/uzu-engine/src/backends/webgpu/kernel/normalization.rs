use super::Shader;
pub(super) fn encode(
    name: &str,
    s: &mut Shader,
) {
    if name == "QKVNorm" {
        s.groups = [s.u("batch_size") * s.u("head_count"), 1, 1];
        s.kernel("qkv_norm", 0);
        if s.flag("in_place") {
            s.alias("qkvg_input", "qkvg_output");
        }
    } else {
        s.groups = [s.u("batch_size"), 1, 1];
        s.kernel(
            if s.u("__subgroup_width") > 0 {
                "normalization_subgroup"
            } else {
                "normalization"
            },
            0,
        );
        if s.flag("in_place") {
            s.alias("input", "output");
        }
    }
}
