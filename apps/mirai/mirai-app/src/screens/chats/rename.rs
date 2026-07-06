use gpui::{Context, FontWeight, IntoElement, black, div, prelude::*, px};

use super::{util::validate_rename_name, view::ChatsView};
use crate::{
    components::{Button, ButtonKind, ButtonSize},
    persistence,
    theme::ActiveTheme,
};

impl ChatsView {
    pub(super) fn open_rename(
        &mut self,
        title: &str,
        cx: &mut Context<Self>,
    ) {
        self.rename_open = true;
        self.rename_error = None;
        self.rename_input.update(cx, |input, cx| input.set_text(title, cx));
        cx.notify();
    }

    fn close_rename(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        self.rename_open = false;
        self.rename_error = None;
        cx.notify();
    }

    pub(super) fn confirm_rename(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        let Some(id) = self.selected.iter().next().cloned().filter(|_| self.selected.len() == 1) else {
            return;
        };
        let text = self.rename_input.read(cx).text();
        match validate_rename_name(&text) {
            Err(msg) => {
                self.rename_error = Some(msg);
                cx.notify();
            },
            Ok(title) => {
                if persistence::rename_chat(&id, &title) {
                    self.close_rename(cx);
                    self.reload(cx);
                } else {
                    self.rename_error = Some("failed to rename chat");
                    cx.notify();
                }
            },
        }
    }

    pub(super) fn rename_modal(
        &self,
        cx: &mut Context<Self>,
    ) -> Option<impl IntoElement> {
        if !self.rename_open {
            return None;
        }
        let theme = cx.theme().clone();
        let error = self.rename_error;

        Some(
            div()
                .absolute()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .bg(black().opacity(0.5))
                .occlude()
                .child(
                    div()
                        .w(px(400.))
                        .flex()
                        .flex_col()
                        .gap_3()
                        .p_4()
                        .rounded_xl()
                        .bg(theme.card)
                        .border_1()
                        .border_color(theme.border)
                        .child(div().text_color(theme.text).font_weight(FontWeight::MEDIUM).child("Rename chat"))
                        .child(self.rename_input.clone())
                        .children(error.map(|msg| div().text_xs().text_color(theme.error).child(msg)))
                        .child(
                            div()
                                .flex()
                                .justify_end()
                                .gap_2()
                                .child(
                                    Button::new("rename-cancel", "Cancel")
                                        .kind(ButtonKind::Secondary)
                                        .size(ButtonSize::Small)
                                        .on_click(cx.listener(|this, _, _, cx| {
                                            this.close_rename(cx);
                                        })),
                                )
                                .child(
                                    Button::new("rename-save", "Save")
                                        .kind(ButtonKind::Primary)
                                        .size(ButtonSize::Small)
                                        .on_click(cx.listener(|this, _, _, cx| {
                                            this.confirm_rename(cx);
                                        })),
                                ),
                        ),
                ),
        )
    }
}
