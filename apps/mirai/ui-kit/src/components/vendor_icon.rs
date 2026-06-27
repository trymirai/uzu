//! Vendor badge: the provider's remote logo when available (loaded by URL from
//! the engine's model metadata), falling back to the vendor's initial on a
//! deterministic per-vendor color. The colored initial is always rendered as a
//! base, so offline / before-load it shows a neutral stand-in (no bundled
//! third-party brand assets).

use gpui::{App, FontWeight, IntoElement, RenderOnce, SharedString, Window, div, hsla, img, prelude::*, px, white};

#[derive(IntoElement)]
pub struct VendorIcon {
    vendor: SharedString,
    size: f32,
    icon_url: Option<SharedString>,
}

impl VendorIcon {
    pub fn new(vendor: impl Into<SharedString>) -> Self {
        Self {
            vendor: vendor.into(),
            size: 20.0,
            icon_url: None,
        }
    }

    pub fn size(
        mut self,
        size: f32,
    ) -> Self {
        self.size = size;
        self
    }

    /// Remote logo URL (from the model's vendor/family metadata). When set and
    /// reachable, it's overlaid on the fallback initial.
    pub fn icon_url(
        mut self,
        url: Option<impl Into<SharedString>>,
    ) -> Self {
        self.icon_url = url.map(Into::into);
        self
    }
}

impl RenderOnce for VendorIcon {
    fn render(
        self,
        _window: &mut Window,
        _cx: &mut App,
    ) -> impl IntoElement {
        let name = self.vendor.trim();
        let size = self.size;
        let base = div().w(px(size)).h(px(size)).flex_none().rounded_md().overflow_hidden();

        match self.icon_url {
            // Remote logo: the brand asset has its own background — no tint behind it.
            Some(url) => base.child(img(url).size_full()),
            None => {
                let initial = name
                    .chars()
                    .find(|c| c.is_alphanumeric())
                    .map(|c| c.to_ascii_uppercase().to_string())
                    .unwrap_or_else(|| "?".to_string());
                base.bg(hsla(vendor_hue(name), 0.5, 0.45, 1.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_color(white())
                    .text_size(px(size * 0.5))
                    .font_weight(FontWeight::SEMIBOLD)
                    .child(initial)
            },
        }
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
