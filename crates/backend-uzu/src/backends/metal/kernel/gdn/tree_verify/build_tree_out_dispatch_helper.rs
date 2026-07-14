#![allow(dead_code)]

use crate::backends::metal::device_tier::DeviceTier;

pub(crate) fn build_tree_out_transposed_h0(
    use_mxu: bool,
    device_tier: DeviceTier,
    use_h0: bool,
) -> bool {
    use_h0 && !use_mxu && matches!(device_tier, DeviceTier::SmallLegacy | DeviceTier::SmallApple8)
}
