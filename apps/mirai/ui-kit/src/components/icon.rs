use gpui::{App, Hsla, IntoElement, RenderOnce, Styled, Window, px, svg};

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
    Lightning,
    Lock,
    WifiOff,
    ModelMenu,
    Performance,
    Thinking,
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
            Icon::Lightning => "icons/lightning.svg",
            Icon::Lock => "icons/lock.svg",
            Icon::WifiOff => "icons/wifi-off.svg",
            Icon::ModelMenu => "icons/model-menu.svg",
            Icon::Performance => "icons/performance.svg",
            Icon::Thinking => "icons/thinking.svg",
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

#[derive(IntoElement)]
pub struct IconEl {
    icon: Icon,
    size: f32,
    color: Hsla,

    rotate_deg: f32,
}

impl IconEl {
    pub fn new(
        icon: Icon,
        color: Hsla,
    ) -> Self {
        Self {
            icon,
            size: 16.0,
            color,
            rotate_deg: 0.0,
        }
    }

    pub fn size(
        mut self,
        size: f32,
    ) -> Self {
        self.size = size;
        self
    }

    pub fn rotate(
        mut self,
        deg: f32,
    ) -> Self {
        self.rotate_deg = deg;
        self
    }
}

impl RenderOnce for IconEl {
    fn render(
        self,
        _window: &mut Window,
        _cx: &mut App,
    ) -> impl IntoElement {
        let mut el = svg().path(self.icon.path()).w(px(self.size)).h(px(self.size)).flex_none().text_color(self.color);
        if self.rotate_deg != 0.0 {
            el = el.with_transformation(gpui::Transformation::rotate(gpui::radians(self.rotate_deg.to_radians())));
        }
        el
    }
}
