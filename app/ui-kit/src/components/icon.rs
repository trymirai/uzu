//! Monochrome SVG icons. GPUI paints an SVG as an alpha mask tinted by the
//! element's `text_color`, so an `Icon` just names a bundled svg and a color.

use gpui::{App, Hsla, IntoElement, RenderOnce, Styled, Window, px, svg};

/// The bundled icon set. Each maps to a file under `assets/icons/`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Icon {
    Logo,
    Plus,
    Chats,
    Models,
    Routers,
    Speech,
    Apps,
    Settings,
    Eject,
    Send,
    Stop,
    Search,
    Copy,
    Check,
    ChevronDown,
    ChevronUp,
    ChevronRight,
    ChevronLeft,
    Close,
    Trash,
    Rename,
    Download,
    Pause,
    SidebarToggle,
    Lightning,
    Lock,
    WifiOff,
    ModelMenu,
    Performance,
    Thinking,
    Error,
    Info,
    Code,
    Devices,
    Heart,
    Shield,
    Github,
    XSocial,
    Discord,
    Spinner,
}

impl Icon {
    pub fn path(self) -> &'static str {
        match self {
            Icon::Logo => "icons/logo.svg",
            Icon::Plus => "icons/plus.svg",
            Icon::Chats => "icons/chats.svg",
            Icon::Models => "icons/models.svg",
            Icon::Routers => "icons/routers.svg",
            Icon::Speech => "icons/speech.svg",
            Icon::Apps => "icons/apps.svg",
            Icon::Settings => "icons/settings.svg",
            Icon::Eject => "icons/eject.svg",
            Icon::Send => "icons/send.svg",
            Icon::Stop => "icons/stop.svg",
            Icon::Search => "icons/search.svg",
            Icon::Copy => "icons/copy.svg",
            Icon::Check => "icons/check.svg",
            Icon::ChevronDown => "icons/chevron-down.svg",
            Icon::ChevronUp => "icons/chevron-up.svg",
            Icon::ChevronRight => "icons/chevron-right.svg",
            Icon::ChevronLeft => "icons/chevron-left.svg",
            Icon::Close => "icons/close.svg",
            Icon::Trash => "icons/trash.svg",
            Icon::Rename => "icons/rename.svg",
            Icon::Download => "icons/download.svg",
            Icon::Pause => "icons/pause.svg",
            Icon::SidebarToggle => "icons/sidebar-toggle.svg",
            Icon::Lightning => "icons/lightning.svg",
            Icon::Lock => "icons/lock.svg",
            Icon::WifiOff => "icons/wifi-off.svg",
            Icon::ModelMenu => "icons/model-menu.svg",
            Icon::Performance => "icons/performance.svg",
            Icon::Thinking => "icons/thinking.svg",
            Icon::Error => "icons/error.svg",
            Icon::Info => "icons/info.svg",
            Icon::Code => "icons/code.svg",
            Icon::Devices => "icons/devices.svg",
            Icon::Heart => "icons/heart.svg",
            Icon::Shield => "icons/shield.svg",
            Icon::Github => "icons/github.svg",
            Icon::XSocial => "icons/x.svg",
            Icon::Discord => "icons/discord.svg",
            Icon::Spinner => "icons/spinner.svg",
        }
    }
}

/// A renderable icon element. Defaults to a 16px glyph; color must be supplied
/// (typically `cx.theme().text` / `text_muted`).
#[derive(IntoElement)]
pub struct IconEl {
    icon: Icon,
    size: f32,
    color: Hsla,
}

impl IconEl {
    pub fn new(icon: Icon, color: Hsla) -> Self {
        Self {
            icon,
            size: 16.0,
            color,
        }
    }

    pub fn size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }
}

impl RenderOnce for IconEl {
    fn render(self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        svg()
            .path(self.icon.path())
            .w(px(self.size))
            .h(px(self.size))
            .flex_none()
            .text_color(self.color)
    }
}
