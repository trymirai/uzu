//! A centered confirmation modal (dim backdrop + card with Cancel/Confirm).
//! The owning view holds the "is it open" state and renders this as an overlay.

use gpui::{
    App, ClickEvent, FontWeight, IntoElement, RenderOnce, SharedString, Window, div, prelude::*, px,
};

use crate::{
    components::{Button, ButtonKind, ButtonSize},
    theme::ActiveTheme,
};

type Handler = Box<dyn Fn(&ClickEvent, &mut Window, &mut App) + 'static>;

#[derive(IntoElement)]
pub struct ConfirmModal {
    title: SharedString,
    message: SharedString,
    confirm_label: SharedString,
    danger: bool,
    on_confirm: Option<Handler>,
    on_cancel: Option<Handler>,
}

impl ConfirmModal {
    pub fn new(title: impl Into<SharedString>, message: impl Into<SharedString>) -> Self {
        Self {
            title: title.into(),
            message: message.into(),
            confirm_label: "Confirm".into(),
            danger: false,
            on_confirm: None,
            on_cancel: None,
        }
    }

    pub fn confirm_label(mut self, label: impl Into<SharedString>) -> Self {
        self.confirm_label = label.into();
        self
    }

    pub fn danger(mut self, danger: bool) -> Self {
        self.danger = danger;
        self
    }

    pub fn on_confirm(mut self, handler: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static) -> Self {
        self.on_confirm = Some(Box::new(handler));
        self
    }

    pub fn on_cancel(mut self, handler: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static) -> Self {
        self.on_cancel = Some(Box::new(handler));
        self
    }
}

impl RenderOnce for ConfirmModal {
    fn render(self, _window: &mut Window, cx: &mut App) -> impl IntoElement {
        let theme = cx.theme().clone();

        let mut cancel = Button::new("modal-cancel", "Cancel")
            .kind(ButtonKind::Secondary)
            .size(ButtonSize::Small);
        if let Some(handler) = self.on_cancel {
            cancel = cancel.on_click(move |event, window, cx| handler(event, window, cx));
        }

        let mut confirm = Button::new("modal-confirm", self.confirm_label)
            .kind(if self.danger {
                ButtonKind::Danger
            } else {
                ButtonKind::Primary
            })
            .size(ButtonSize::Small);
        if let Some(handler) = self.on_confirm {
            confirm = confirm.on_click(move |event, window, cx| handler(event, window, cx));
        }

        div()
            .absolute()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(gpui::black().opacity(0.5))
            .occlude() // block interaction with the screen behind
            .child(
                div()
                    .w(px(360.))
                    .flex()
                    .flex_col()
                    .gap_3()
                    .p_4()
                    .rounded_xl()
                    .bg(theme.card)
                    .border_1()
                    .border_color(theme.border)
                    .child(
                        div()
                            .text_color(theme.text)
                            .font_weight(FontWeight::MEDIUM)
                            .child(self.title),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.text_muted)
                            .child(self.message),
                    )
                    .child(
                        div()
                            .flex()
                            .justify_end()
                            .gap_2()
                            .child(cancel)
                            .child(confirm),
                    ),
            )
    }
}
