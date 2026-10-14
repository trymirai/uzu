use super::Shader;
pub(super) fn encode(
    name: &str,
    s: &mut Shader,
) {
    if name == "Softmax" {
        s.groups = [s.u("outer_dim") * s.u("batch_dim"), 1, 1];
        s.kernel("softmax", 0);
        return;
    }
    let (length, kernel) = match name {
        "Activation" => (s.u("n"), "activation"),
        "TensorAddBias" => (s.u("length"), "add_bias"),
        "TensorAddScale" => (s.u("length"), "add_scale"),
        "LogitTransform" => (s.u("length"), "logit_transform"),
        "SigmoidGate" => (s.u("gate_dim") * s.u("batch_dim"), "sigmoid_gate"),
        _ => unreachable!("unknown elementwise kernel {name}"),
    };
    s.set("length_total", length);
    s.groups = [length.div_ceil(256), 1, 1];
    s.kernel(kernel, 0);
    if s.flag("in_place") {
        s.alias("input", "output");
    }
}
