use std::{borrow::Cow, sync::Arc};

use gpui::{App, Context, Global, Hsla, Subscription, rgb, rgba};

pub const FONT_SANS: &str = "Inter Variable";
pub const FONT_MONO: &str = "Geist Mono";

pub const TEXT_SIZE: f32 = 14.0;

pub use crate::tokens::layout;

#[inline]
fn hex(c: u32) -> Hsla {
    rgb(c).into()
}

#[inline]
fn hexa(c: u32) -> Hsla {
    rgba(c).into()
}

#[derive(Clone)]
pub struct Theme {
    pub dark: bool,

    pub bg: Hsla,
    pub bg_sidebar: Hsla,
    pub bg_hover: Hsla,
    pub bg_sub: Hsla,
    pub bg_sub_hover: Hsla,
    pub card: Hsla,

    pub text: Hsla,
    pub text_muted: Hsla,
    pub text_inverse: Hsla,

    pub border: Hsla,
    pub button_border: Hsla,
    pub button_border_hover: Hsla,

    pub accent: Hsla,
    pub success: Hsla,
    pub error: Hsla,
    pub info: Hsla,
}

impl Theme {
    pub fn dark() -> Self {
        Self {
            dark: true,
            bg: hex(0x0B0B0B),
            bg_sidebar: hex(0x080808),
            bg_hover: hex(0x191919),
            bg_sub: hex(0x222222),
            bg_sub_hover: hex(0x313131),
            card: hex(0x0F0F0F),
            text: hex(0xEEEEEE),
            text_muted: hex(0x7B7B7B),
            text_inverse: hex(0x0F0F0F),
            border: hexa(0xFFFFFF12),
            button_border: hexa(0xFFFFFF26),
            button_border_hover: hexa(0xFFFFFF38),
            accent: hex(0xFF6A20),
            success: hex(0x00C758),
            error: hex(0xFB2C36),
            info: hex(0x3080FF),
        }
    }

    pub fn light() -> Self {
        Self {
            dark: false,
            bg: hex(0xFCFCFC),
            bg_sidebar: hex(0xF5F5F5),
            bg_hover: hex(0xF0F0F0),
            bg_sub: hex(0xF0F0F0),
            bg_sub_hover: hex(0xE8E8E8),
            card: hex(0xFFFFFF),
            text: hex(0x202020),
            text_muted: hex(0x838383),
            text_inverse: hex(0xFFFFFF),
            border: hexa(0x0000001A),
            button_border: hexa(0x00000026),
            button_border_hover: hexa(0x00000038),
            accent: hex(0xFF6A20),
            success: hex(0x00C758),
            error: hex(0xFB2C36),
            info: hex(0x3080FF),
        }
    }
}

struct GlobalTheme(Arc<Theme>);

impl Global for GlobalTheme {}

pub trait ActiveTheme {
    fn theme(&self) -> &Arc<Theme>;
}

impl ActiveTheme for App {
    fn theme(&self) -> &Arc<Theme> {
        &self.global::<GlobalTheme>().0
    }
}

pub fn init(cx: &mut App) {
    register_fonts(cx);
    cx.set_global(GlobalTheme(Arc::new(Theme::dark())));
}

pub fn set_theme(
    cx: &mut App,
    theme: Theme,
) {
    cx.set_global(GlobalTheme(Arc::new(theme)));
}

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
