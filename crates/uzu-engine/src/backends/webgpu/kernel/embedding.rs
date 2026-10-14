use super::Shader;
pub(super) fn encode(s: &mut Shader) {
    if s.u("table_kind") == 0 {
        s.set("group_size", 1);
    }
    s.groups = [s.u("model_dim").div_ceil(256), s.u("batch_size"), 1];
    s.kernel(
        if s.u("__subgroup_width") > 0 {
            "embedding_subgroup"
        } else {
            "embedding"
        },
        0,
    );
}
