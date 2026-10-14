use super::Shader;
pub(super) fn encode(
    name: &str,
    s: &mut Shader,
) {
    let tiles = match name {
        "ShortConvPack" => s
            .u("state_stride")
            .checked_add(s.u("suffix_len"))
            .expect("short convolution row count exceeds u32")
            .div_ceil(4)
            .max(1),
        "ShortConvPrefill" => s.u("suffix_len").div_ceil(4).max(1),
        _ => 1,
    };
    s.groups = [s.u("model_dim").div_ceil(128), tiles.min(65535), tiles.div_ceil(65535)];
    s.kernel(
        match name {
            "ShortConvPack" => "short_conv_pack",
            "ShortConvPrefill" => "short_conv_prefill",
            "ShortConvDecode" => "short_conv_decode",
            "ShortConvTrie" => "short_conv_trie",
            _ => unreachable!(),
        },
        0,
    );
}
