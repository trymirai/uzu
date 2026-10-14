//! Gated DeltaNet scheduling. Arithmetic is compiled from the Slang kernels.
use super::Shader;
pub(super) fn encode(
    name: &str,
    s: &mut Shader,
) {
    let (kernel, width) = match name {
        "Conv1dPack" => {
            s.groups = [((s.u("state_stride") + s.u("suffix_len")) * s.u("num_channels")).div_ceil(256), 1, 1];
            ("gdn_conv_pack", 0)
        },
        "DeltaNetConvUpdate" => {
            s.groups = [s.u("conv_dim").div_ceil(256), 1, 1];
            ("gdn_conv_update", 0)
        },
        "DeltaNetConvScan" => {
            let tiles = s.u("suffix_len").div_ceil(4).max(1);
            s.groups = [s.u("conv_dim").div_ceil(128), tiles.min(65535), tiles.div_ceil(65535)];
            ("gdn_conv_scan", 0)
        },
        "DeltaNetNormGate" => {
            s.groups = [s.u("num_v_heads"), s.u("suffix_len"), 1];
            ("gdn_norm_gate", 0)
        },
        "DeltaNetPrefillPrep" => {
            let width = s.u("HEAD_K_DIM").next_power_of_two();
            s.set("width", width);
            s.groups = [s.u("num_k_heads"), s.u("suffix_len"), 1];
            ("gdn_prefill_prep", width)
        },
        "DeltaNetPrefill" => {
            // Each workgroup owns distinct value rows of the F32 state.
            let subgroup_width = s.u("__subgroup_width");
            if subgroup_width > 0 {
                s.groups = [s.u("head_v_dim").div_ceil(256 / subgroup_width), s.u("num_v_heads"), 1];
                ("gdn_prefill_subgroup", subgroup_width)
            } else {
                s.groups = [s.u("head_v_dim").div_ceil(32), s.u("num_v_heads"), 1];
                ("gdn_prefill", s.u("HEAD_K_DIM").next_power_of_two())
            }
        },
        "DeltaNetUpdate" => {
            // Each value head has one workgroup and disjoint state elements.
            let width = s.u("HEAD_K_DIM").max(s.u("head_v_dim")).next_power_of_two().max(32);
            assert!(width <= 256, "WebGPU DeltaNet head dimensions must not exceed 256");
            s.set("width", width);
            s.groups = [s.u("num_v_heads"), 1, 1];
            if s.u("__subgroup_width") > 0 {
                ("gdn_update_subgroup", s.u("__subgroup_width"))
            } else {
                ("gdn_update", width)
            }
        },
        "ConvTreeScan" => {
            s.groups = [s.u("total_proj_dim").div_ceil(128), s.u("suffix_len"), 1];
            ("gdn_conv_tree_scan", 0)
        },
        "StateAdvance" => {
            s.groups = [s.u("HEAD_K_DIM").div_ceil(32), s.u("num_v_heads"), 1];
            ("gdn_state_advance", s.u("HEAD_K_DIM").next_power_of_two())
        },
        _ => unreachable!(),
    };
    s.kernel(kernel, width);
}
