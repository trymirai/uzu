use gpui::{
    App, ClickEvent, CursorStyle, ElementId, FontWeight, IntoElement, RenderOnce, SharedString, Window, div,
    prelude::*, px,
};

use crate::{
    components::icon::{Icon, IconEl},
    theme::ActiveTheme,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ButtonKind {
    Primary,
    Secondary,
    Danger,
    Ghost,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ButtonSize {
    Small,
    Medium,
}

#[derive(IntoElement)]
pub struct Button {
    id: ElementId,
    label: SharedString,
    kind: ButtonKind,
    size: ButtonSize,
    icon: Option<Icon>,
    disabled: bool,
    full_width: bool,
    on_click: Option<Box<dyn Fn(&ClickEvent, &mut Window, &mut App) + 'static>>,
}

impl Button {
    pub fn new(
        id: impl Into<ElementId>,
        label: impl Into<SharedString>,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            kind: ButtonKind::Secondary,
            size: ButtonSize::Medium,
            icon: None,
            disabled: false,
            full_width: false,
            on_click: None,
        }
    }

    pub fn kind(
        mut self,
        kind: ButtonKind,
    ) -> Self {
        self.kind = kind;
        self
    }

    pub fn size(
        mut self,
        size: ButtonSize,
    ) -> Self {
        self.size = size;
        self
    }

    pub fn icon(
        mut self,
        icon: Icon,
    ) -> Self {
        self.icon = Some(icon);
        self
    }

    pub fn disabled(
        mut self,
        disabled: bool,
    ) -> Self {
        self.disabled = disabled;
        self
    }

    pub fn full_width(
        mut self,
        full_width: bool,
    ) -> Self {
        self.full_width = full_width;
        self
    }

    pub fn on_click(
        mut self,
        handler: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.on_click = Some(Box::new(handler));
        self
    }
}

impl RenderOnce for Button {
    fn render(
        self,
        _window: &mut Window,
        cx: &mut App,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();

        let (bg, fg, border) = match self.kind {
            ButtonKind::Primary => (theme.text, theme.bg, None),
            ButtonKind::Secondary => (gpui::transparent_black(), theme.text, Some(theme.button_border)),
            ButtonKind::Danger => (theme.error, gpui::white(), None),
            ButtonKind::Ghost => (gpui::transparent_black(), theme.text_muted, None),
        };
        let hover_bg = match self.kind {
            ButtonKind::Primary => theme.text.opacity(0.85),
            ButtonKind::Danger => theme.error.opacity(0.85),
            _ => theme.bg_hover,
        };
        let (height, text_size) = match self.size {
            ButtonSize::Small => (28.0_f32, 13.0_f32),
            ButtonSize::Medium => (36.0, 14.0),
        };

        let mut el = div()
            .id(self.id)
            .flex()
            .items_center()
            .justify_center()
            .gap_2()
            .h(px(height))
            .px_3()
            .bg(bg)
            .text_color(fg)
            .text_size(px(text_size))
            .font_weight(FontWeight::MEDIUM);

        el = match self.size {
            ButtonSize::Small => el.rounded_md(),
            ButtonSize::Medium => el.rounded_lg(),
        };
        if self.full_width {
            el = el.w_full();
        }
        if let Some(border) = border {
            el = el.border_1().border_color(border);
        }
        if let Some(icon) = self.icon {
            el = el.child(IconEl::new(icon, fg).size(text_size + 2.0));
        }
        el = el.child(self.label);

        if self.disabled {
            el.opacity(0.5)
        } else {
            let mut el = el.cursor(CursorStyle::PointingHand).hover(move |s| s.bg(hover_bg));
            if let Some(handler) = self.on_click {
                el = el.on_click(move |event, window, cx| handler(event, window, cx));
            }
            el
        }
    }
}
