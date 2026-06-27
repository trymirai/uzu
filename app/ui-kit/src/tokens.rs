//! App-wide design tokens: font-size, icon-size, radius, and layout scales.
//!
//! Colors live in [`crate::theme`] (the one place that varies by light/dark).
//! Sizes are constant across themes, so they're compile-time `const`s — zero
//! runtime cost and no `Theme` bloat.
//!
//! GPUI's Tailwind-style helpers (`.gap_2`, `.px_3`, `.rounded_lg`) are already
//! a named scale and are left as-is. These tokens cover only the raw literals
//! that bypass a scale: `.text_size(px(..))`, numeric icon `.size(n.)`, and
//! `.rounded(px(..))`.

use gpui::{Pixels, px};

/// Font sizes (logical px) for `.text_size(..)`. A typographic scale; replaces
/// inline `px(..)` so body/caption/heading sizes live in one place.
pub mod font {
    use super::*;
    /// Smallest labels — code-block language tag, perf captions. 11px.
    pub const CAPTION: Pixels = px(11.0);
    /// Secondary small text. 12px.
    pub const SMALL: Pixels = px(12.0);
    /// Compact text (dense rows, monospace meta). 13px.
    pub const COMPACT: Pixels = px(13.0);
    /// Base UI/body text. 14px (mirrors [`crate::theme::TEXT_SIZE`]).
    pub const BODY: Pixels = px(14.0);
    /// Section labels / emphasized body. 16px.
    pub const LABEL: Pixels = px(16.0);
    /// Markdown H3. 18px.
    pub const H3: Pixels = px(18.0);
    /// Markdown H2. 20px.
    pub const H2: Pixels = px(20.0);
    /// Prominent heading / hero subtitle. 22px.
    pub const HEADING: Pixels = px(22.0);
    /// Markdown H1. 24px.
    pub const H1: Pixels = px(24.0);
    /// Display / welcome title. 32px.
    pub const DISPLAY: Pixels = px(32.0);
}

/// Icon glyph sizes (`f32`, matching `IconEl::size(f32)` / `VendorIcon::size`).
pub mod icon {
    /// Inline/compact glyphs. 12px.
    pub const XS: f32 = 12.0;
    /// Small controls. 14px.
    pub const SM: f32 = 14.0;
    /// Default icon size. 16px.
    pub const MD: f32 = 16.0;
    /// Header/section icons. 18px.
    pub const LG: f32 = 18.0;
    /// Large header icons. 20px.
    pub const XL: f32 = 20.0;
    /// Prominent vendor marks. 22px.
    pub const XXL: f32 = 22.0;
    /// Hero/empty-state logo. 64px.
    pub const HERO: f32 = 64.0;
}

/// Corner radii (logical px) for `.rounded(..)`. Tailwind `.rounded_lg`/`_md`
/// helpers are unaffected; this covers the raw `rounded(px(..))` cases.
pub mod radius {
    use super::*;
    /// Small chips/inputs. 4px.
    pub const SM: Pixels = px(4.0);
    /// Buttons/cells. 8px (== legacy `layout::RADIUS`).
    pub const MD: Pixels = px(8.0);
}

/// Layout dimensions (logical px, `f32`; wrap with `px(..)` where `Pixels` is
/// required). Migrated from the former `theme::layout`.
pub mod layout {
    /// Left navigation sidebar width (desktop).
    pub const SIDEBAR_WIDTH: f32 = 200.0;
    /// Bottom footer bar height.
    pub const FOOTER_HEIGHT: f32 = 24.0;
    /// Max width of the centered content column (chat / history / settings).
    pub const CONTENT_MAX_WIDTH: f32 = 800.0;
    /// Settings vertical tab sidebar width.
    pub const SETTINGS_SIDEBAR_WIDTH: f32 = 152.0;
    /// Default corner radius for buttons/cells (legacy alias of [`super::radius::MD`]).
    pub const RADIUS: f32 = 8.0;
}
