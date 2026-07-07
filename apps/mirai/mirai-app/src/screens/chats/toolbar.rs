use gpui::{AnyElement, Context, CursorStyle, FontWeight, IntoElement, div, prelude::*, px};

use super::view::ChatsView;
use crate::{
    components::{Button, ButtonKind, ButtonSize, Icon, IconButton, IconEl},
    theme::ActiveTheme,
    tokens,
};

impl ChatsView {
    pub(super) fn toolbar(
        &self,
        cx: &mut Context<Self>,
        filtered: Vec<String>,
        empty: bool,
    ) -> AnyElement {
        let theme = cx.theme().clone();

        if self.selection_mode {
            let total = filtered.len();
            let count = self.selected.len();
            let all = total > 0 && count == total;

            let mut select_all = div()
                .id("select-all")
                .size(px(18.))
                .flex_none()
                .rounded(tokens::radius::SM)
                .border_1()
                .border_color(theme.border)
                .flex()
                .items_center()
                .justify_center()
                .cursor(CursorStyle::PointingHand)
                .on_click(cx.listener(move |this, _, _, cx| {
                    if this.selected.len() == filtered.len() && !filtered.is_empty() {
                        this.selected.clear();
                    } else {
                        this.selected = filtered.iter().cloned().collect();
                    }
                    cx.notify();
                }));
            if all {
                select_all =
                    select_all.bg(theme.info).child(IconEl::new(Icon::Check, theme.card).size(tokens::icon::XS));
            }

            let show_rename = count == 1 && !all;

            let mut actions = div().flex().items_center().gap_2();
            if show_rename {
                let title = self
                    .chats
                    .iter()
                    .find(|c| self.selected.contains(&c.id))
                    .map(|c| c.title.clone())
                    .unwrap_or_default();
                actions = actions.child(IconButton::new("rename-one", Icon::Rename).color(theme.text_muted).on_click(
                    cx.listener(move |this, _, _, cx| {
                        this.open_rename(&title, cx);
                    }),
                ));
            }
            actions = actions
                .child(
                    Button::new("bulk-delete", "Delete")
                        .kind(ButtonKind::Danger)
                        .size(ButtonSize::Small)
                        .disabled(count == 0)
                        .on_click(cx.listener(|this, _, _, cx| {
                            if !this.selected.is_empty() {
                                this.confirm_bulk_delete = true;
                                cx.notify();
                            }
                        })),
                )
                .child(IconButton::new("sel-exit", Icon::Close).color(theme.text_muted).on_click(cx.listener(
                    |this, _, _, cx| {
                        this.selection_mode = false;
                        this.selected.clear();
                        cx.notify();
                    },
                )));

            div()
                .flex()
                .items_center()
                .justify_between()
                .mb_3()
                .child(div().flex().items_center().gap_3().child(select_all).child(
                    div().text_sm().text_color(theme.text_muted).child(if count == 0 {
                        "Select all".to_string()
                    } else if all {
                        "All selected".to_string()
                    } else {
                        format!("{count} selected")
                    }),
                ))
                .child(actions)
                .into_any_element()
        } else {
            div()
                .flex()
                .items_center()
                .gap_3()
                .mb_3()
                .child(div().text_sm().font_weight(FontWeight::MEDIUM).text_color(theme.text).child("Your chats"))
                .child(
                    div()
                        .flex_1()
                        .flex()
                        .items_center()
                        .gap_2()
                        .px_3()
                        .py_2()
                        .rounded_lg()
                        .border_1()
                        .border_color(theme.border)
                        .bg(theme.card)
                        .child(IconEl::new(Icon::Search, theme.text_muted).size(tokens::icon::SM))
                        .child(self.search.clone()),
                )
                .child(
                    Button::new("select-mode", "Select")
                        .kind(ButtonKind::Secondary)
                        .size(ButtonSize::Small)
                        .disabled(empty)
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.selection_mode = true;
                            this.selected.clear();
                            cx.notify();
                        })),
                )
                .into_any_element()
        }
    }
}
