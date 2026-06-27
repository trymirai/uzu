//! App-wide design tokens: font-size, icon-size, radius, and layout scales.
//!
//! Colors live in [`crate::theme`] (the only theme-dependent values); sizes are
//! constant, so they're compile-time `const`s. GPUI's Tailwind helpers
//! (`.gap_2`, `.rounded_lg`, …) are already a scale and stay as-is — these cover
//! only the raw literals that bypass one: `.text_size(px(..))`, numeric icon
//! `.size(n.)`, and `.rounded(px(..))`.

use gpui::{Pixels, px};

/// Font sizes for `.text_size(..)`.
pub mod font {
    use super::*;
    pub const CAPTION: Pixels = px(11.0);
    pub const SMALL: Pixels = px(12.0);
    pub const COMPACT: Pixels = px(13.0);
    pub const BODY: Pixels = px(14.0);
    pub const LABEL: Pixels = px(16.0);
    pub const H3: Pixels = px(18.0);
    pub const H2: Pixels = px(20.0);
    pub const HEADING: Pixels = px(22.0);
    pub const H1: Pixels = px(24.0);
    pub const DISPLAY: Pixels = px(32.0);
}

/// Icon glyph sizes (`f32`, matching `IconEl::size`).
pub mod icon {
    pub const XS: f32 = 12.0;
    pub const SM: f32 = 14.0;
    pub const MD: f32 = 16.0;
    pub const LG: f32 = 18.0;
    pub const XL: f32 = 20.0;
    pub const XXL: f32 = 22.0;
    pub const HERO: f32 = 64.0;
}

/// Corner radii for `.rounded(..)` (Tailwind `.rounded_*` helpers stay as-is).
pub mod radius {
    use super::*;
    pub const SM: Pixels = px(4.0);
}

/// Layout dimensions (`f32`; wrap with `px(..)` where `Pixels` is needed).
pub mod layout {
    pub const SIDEBAR_WIDTH: f32 = 200.0;
    pub const FOOTER_HEIGHT: f32 = 24.0;
    pub const CONTENT_MAX_WIDTH: f32 = 800.0;
}
