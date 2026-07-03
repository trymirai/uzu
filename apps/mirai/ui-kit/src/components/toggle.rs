use gpui::{App, ClickEvent, CursorStyle, ElementId, IntoElement, RenderOnce, Window, div, prelude::*, px};

use crate::theme::ActiveTheme;

#[derive(IntoElement)]
pub struct Toggle {
    id: ElementId,
    on: bool,
    on_click: Option<Box<dyn Fn(&ClickEvent, &mut Window, &mut App) + 'static>>,
}

impl Toggle {
    pub fn new(
        id: impl Into<ElementId>,
        on: bool,
    ) -> Self {
        Self {
            id: id.into(),
            on,
            on_click: None,
        }
    }

    pub fn on_click(
        mut self,
        handler: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.on_click = Some(Box::new(handler));
        self
    }
}

impl RenderOnce for Toggle {
    fn render(
        self,
        _window: &mut Window,
        cx: &mut App,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();

        let on_track = if theme.dark {
            gpui::white()
        } else {
            theme.text
        };
        let track = if self.on {
            on_track
        } else {
            theme.bg_sub_hover
        };
        let knob = if self.on {
            theme.bg
        } else {
            gpui::white()
        };
        let mut el = div()
            .id(self.id)
            .w(px(40.))
            .h(px(22.))
            .rounded_full()
            .bg(track)
            .relative()
            .cursor(CursorStyle::PointingHand)
            .child(
                div()
                    .absolute()
                    .top(px(2.))
                    .left(px(if self.on {
                        20.
                    } else {
                        2.
                    }))
                    .size(px(18.))
                    .rounded_full()
                    .bg(knob),
            );
        if let Some(handler) = self.on_click {
            el = el.on_click(move |event, window, cx| handler(event, window, cx));
        }
        el
    }
}
