use super::Shader;
pub(super) fn encode(
    name: &str,
    s: &mut Shader,
) {
    let gated = name == "GatedActMul";
    let columns = s.u(if gated {
        "gated_dim"
    } else {
        "element_count"
    });
    let rows = s.u(if gated {
        "batch_dim"
    } else {
        "batch_size"
    });
    s.set("columns", columns);
    s.groups = [columns.div_ceil(256), rows, 1];
    let kernel = match (gated, s.u("__subgroup_width") > 0) {
        (true, true) => "gated_act_mul_subgroup",
        (true, false) => "gated_act_mul",
        (false, true) => "activation_transform_subgroup",
        (false, false) => "activation_transform",
    };
    s.kernel(kernel, 0);
    if s.flag("in_place") {
        s.alias("input", "fp_out");
    }
}
