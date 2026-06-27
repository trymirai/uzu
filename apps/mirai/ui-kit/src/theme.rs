//! Theme tokens (ported from `@trymirai-schemas/ui-kit`), bundled fonts, layout
//! constants, and a `cx.theme()` accessor. Dark by default.

use std::{borrow::Cow, sync::Arc};

use gpui::{App, Context, Global, Hsla, Subscription, rgb, rgba};

/// Family names embedded in the `.ttf` files (not the file names); variable fonts.
pub const FONT_SANS: &str = "Inter Variable";
pub const FONT_MONO: &str = "Geist Mono";

/// Base UI text size (mirai-chat body text is ~14px).
pub const TEXT_SIZE: f32 = 14.0;

/// Layout constants — now live in [`crate::tokens::layout`]; re-exported here
/// for back-compat (`theme::layout::*` keeps resolving).
pub use crate::tokens::layout;

#[inline]
fn hex(c: u32) -> Hsla {
    rgb(c).into()
}

/// `0xRRGGBBAA` → `Hsla` (for ui-kit's alpha border tokens).
#[inline]
fn hexa(c: u32) -> Hsla {
    rgba(c).into()
}

/// The active color palette. All values are resolved up-front so views can read
/// them with `cx.theme().<field>`.
#[derive(Clone)]
pub struct Theme {
    pub dark: bool,

    pub bg: Hsla,
    pub bg_sidebar: Hsla,
    pub bg_hover: Hsla,
    pub bg_sub: Hsla,
    pub bg_sub_hover: Hsla,
    pub card: Hsla,
    pub card_hover: Hsla,

    pub text: Hsla,
    pub text_muted: Hsla,
    pub text_inverse: Hsla,
    pub link: Hsla,

    pub border: Hsla,
    pub button_border: Hsla,
    pub button_border_hover: Hsla,

    pub accent: Hsla,
    pub success: Hsla,
    pub error: Hsla,
    pub info: Hsla,
}

impl Theme {
    /// Dark palette, mapped to ui-kit tokens (`palette.css` / `semantic.css`).
    pub fn dark() -> Self {
        Self {
            dark: true,
            bg: hex(0x0B0B0B),                     // --color-background
            bg_sidebar: hex(0x080808),             // darker than content (mirai-chat layout)
            bg_hover: hex(0x191919),               // ghost-hover (gray-200)
            bg_sub: hex(0x222222),                 // surface-tertiary (gray-300)
            bg_sub_hover: hex(0x313131),           // gray-500
            card: hex(0x0F0F0F),                   // surface-elevated (gray-50)
            card_hover: hex(0x191919),             // gray-200
            text: hex(0xEEEEEE),                   // text-primary (gray-1200)
            text_muted: hex(0x7B7B7B),             // text-muted (gray-1000)
            text_inverse: hex(0x0F0F0F),           // primary-contrast (gray-50)
            link: hex(0xFF6A20),                   // Mirai accent
            border: hexa(0xFFFFFF12),              // border-default (white @ ~0.07)
            button_border: hexa(0xFFFFFF26),       // border-outlined (white @ ~0.15)
            button_border_hover: hexa(0xFFFFFF38), // border-outlined-hover (~0.22)
            accent: hex(0xFF6A20),
            success: hex(0x00C758), // green-500
            error: hex(0xFB2C36),   // red-500
            info: hex(0x3080FF),    // blue-500
        }
    }

    /// Light palette mapped to ui-kit tokens (light).
    pub fn light() -> Self {
        Self {
            dark: false,
            bg: hex(0xFCFCFC), // --color-background
            bg_sidebar: hex(0xF5F5F5),
            bg_hover: hex(0xF0F0F0),     // ghost-hover (gray-300)
            bg_sub: hex(0xF0F0F0),       // surface-tertiary (gray-300)
            bg_sub_hover: hex(0xE8E8E8), // gray-400
            card: hex(0xFFFFFF),         // surface-elevated (gray-50)
            card_hover: hex(0xF9F9F9),   // gray-200
            text: hex(0x202020),         // text-primary (gray-1200)
            text_muted: hex(0x838383),   // text-muted (gray-1000)
            text_inverse: hex(0xFFFFFF), // primary-contrast (gray-50)
            link: hex(0xFF6A20),
            border: hexa(0x0000001A),              // border-default (black @ ~0.10)
            button_border: hexa(0x00000026),       // border-outlined (black @ ~0.15)
            button_border_hover: hexa(0x00000038), // border-outlined-hover (~0.22)
            accent: hex(0xFF6A20),
            success: hex(0x00C758),
            error: hex(0xFB2C36),
            info: hex(0x3080FF),
        }
    }
}

struct GlobalTheme(Arc<Theme>);

impl Global for GlobalTheme {}

/// `cx.theme()` accessor (mirrors Zed's `ActiveTheme`).
pub trait ActiveTheme {
    fn theme(&self) -> &Arc<Theme>;
}

impl ActiveTheme for App {
    fn theme(&self) -> &Arc<Theme> {
        &self.global::<GlobalTheme>().0
    }
}

/// Register fonts and install the default (dark) theme. Call once at startup.
pub fn init(cx: &mut App) {
    register_fonts(cx);
    cx.set_global(GlobalTheme(Arc::new(Theme::dark())));
}

/// Swap the active palette; notifies observers.
pub fn set_theme(
    cx: &mut App,
    theme: Theme,
) {
    cx.set_global(GlobalTheme(Arc::new(theme)));
}

/// Run `on_change` whenever the palette is swapped.
pub fn observe_theme<V: 'static>(
    cx: &mut Context<V>,
    mut on_change: impl FnMut(&mut V, &mut Context<V>) + 'static,
) -> Subscription {
    cx.observe_global::<GlobalTheme>(move |this, cx| on_change(this, cx))
}

fn register_fonts(cx: &mut App) {
    let fonts: Vec<Cow<'static, [u8]>> = vec![
        Cow::Borrowed(include_bytes!("../assets/fonts/InterVariable.ttf").as_slice()),
        Cow::Borrowed(include_bytes!("../assets/fonts/GeistMono.ttf").as_slice()),
    ];
    if let Err(err) = cx.text_system().add_fonts(fonts) {
        eprintln!("[mirai-app] failed to register bundled fonts: {err}");
    }
}
