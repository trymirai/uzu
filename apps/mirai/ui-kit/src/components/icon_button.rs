use gpui::{App, ClickEvent, CursorStyle, ElementId, Hsla, IntoElement, RenderOnce, Window, div, prelude::*, px};

use super::ClickHandler;
use crate::{
    components::icon::{Icon, IconEl},
    theme::ActiveTheme,
};

#[derive(IntoElement)]
pub struct IconButton {
    id: ElementId,
    icon: Icon,
    icon_size: f32,
    hit_size: f32,
    color: Option<Hsla>,
    background: Option<Hsla>,
    disabled: bool,
    on_click: Option<ClickHandler>,
}

impl IconButton {
    pub fn new(
        id: impl Into<ElementId>,
        icon: Icon,
    ) -> Self {
        Self {
            id: id.into(),
            icon,
            icon_size: 16.0,
            hit_size: 28.0,
            color: None,
            background: None,
            disabled: false,
            on_click: None,
        }
    }

    pub fn color(
        mut self,
        color: Hsla,
    ) -> Self {
        self.color = Some(color);
        self
    }

    pub fn background(
        mut self,
        background: Hsla,
    ) -> Self {
        self.background = Some(background);
        self
    }

    pub fn icon_size(
        mut self,
        size: f32,
    ) -> Self {
        self.icon_size = size;
        self
    }

    pub fn hit_size(
        mut self,
        size: f32,
    ) -> Self {
        self.hit_size = size;
        self
    }

    pub fn disabled(
        mut self,
        disabled: bool,
    ) -> Self {
        self.disabled = disabled;
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

impl RenderOnce for IconButton {
    fn render(
        self,
        _window: &mut Window,
        cx: &mut App,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let color = self.color.unwrap_or(theme.text_muted);

        let hover_bg = if self.background.is_some() {
            theme.bg_sub_hover
        } else {
            theme.bg_hover
        };

        let el = div()
            .id(self.id)
            .flex()
            .items_center()
            .justify_center()
            .size(px(self.hit_size))
            .rounded_md()
            .when_some(self.background, |el, bg| el.bg(bg))
            .child(IconEl::new(self.icon, color).size(self.icon_size));

        if self.disabled {
            el.opacity(0.5)
        } else {
            let mut el = el.cursor(CursorStyle::PointingHand).hover(move |s| s.bg(hover_bg));
            if let Some(handler) = self.on_click {
                el = el.on_click(handler);
            }
            el
        }
    }
}
