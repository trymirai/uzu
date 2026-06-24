//! Vendor badge: the vendor's initial on a deterministic per-vendor color.
//! Used where ui-kit shows a vendor logo — we avoid bundling third-party brand
//! assets, so this is a neutral, distinct-per-vendor stand-in.

use gpui::{
    App, FontWeight, IntoElement, RenderOnce, SharedString, Window, div, hsla, prelude::*, px, white,
};

#[derive(IntoElement)]
pub struct VendorIcon {
    vendor: SharedString,
    size: f32,
}

impl VendorIcon {
    pub fn new(vendor: impl Into<SharedString>) -> Self {
        Self {
            vendor: vendor.into(),
            size: 20.0,
        }
    }

    pub fn size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }
}

impl RenderOnce for VendorIcon {
    fn render(self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        let name = self.vendor.trim();
        let initial = name
            .chars()
            .find(|c| c.is_alphanumeric())
            .map(|c| c.to_ascii_uppercase().to_string())
            .unwrap_or_else(|| "?".to_string());
        div()
            .w(px(self.size))
            .h(px(self.size))
            .flex_none()
            .flex()
            .items_center()
            .justify_center()
            .rounded_md()
            .bg(hsla(vendor_hue(name), 0.5, 0.45, 1.0))
            .text_color(white())
            .text_size(px(self.size * 0.5))
            .font_weight(FontWeight::SEMIBOLD)
            .child(initial)
    }
}

/// Deterministic hue (0–1) from the vendor name (FNV-1a hash).
fn vendor_hue(name: &str) -> f32 {
    let mut h: u32 = 2166136261;
    for b in name.to_lowercase().bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    (h % 360) as f32 / 360.0
}
