use gpui::{Pixels, px};

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

pub mod icon {
    pub const XS: f32 = 12.0;
    pub const SM: f32 = 14.0;
    pub const MD: f32 = 16.0;
    pub const LG: f32 = 18.0;
    pub const XL: f32 = 20.0;
    pub const XXL: f32 = 22.0;
    pub const HERO: f32 = 64.0;
}

pub mod radius {
    use super::*;
    pub const SM: Pixels = px(4.0);
}

pub mod layout {
    pub const SIDEBAR_WIDTH: f32 = 200.0;
    pub const FOOTER_HEIGHT: f32 = 24.0;
    pub const CONTENT_MAX_WIDTH: f32 = 800.0;
}
