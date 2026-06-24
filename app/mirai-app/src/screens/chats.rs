//! Chat history screen: lists saved chats (newest first), open on click, delete
//! per row.

use gpui::{
    Context, CursorStyle, EventEmitter, FontWeight, IntoElement, Render, SharedString, Window, div,
    prelude::*, px,
};

use crate::{
    components::{ConfirmModal, Icon, IconButton},
    persistence::{self, StoredChat},
    theme::{ActiveTheme, layout::CONTENT_MAX_WIDTH},
};

/// Emitted to the shell to open a saved chat.
pub enum ChatsEvent {
    Open(String),
}

pub struct ChatsView {
    chats: Vec<StoredChat>,
    /// (id, title) of a chat pending delete confirmation.
    confirm_delete: Option<(String, String)>,
}

impl EventEmitter<ChatsEvent> for ChatsView {}

impl ChatsView {
    pub fn new(_cx: &mut Context<Self>) -> Self {
        Self {
            chats: persistence::list_chats(),
            confirm_delete: None,
        }
    }

    pub fn reload(&mut self, cx: &mut Context<Self>) {
        self.chats = persistence::list_chats();
        cx.notify();
    }

    fn row(&self, cx: &mut Context<Self>, chat: &StoredChat) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover_bg = theme.bg_hover;
        let open_id = chat.id.clone();
        let delete_id = chat.id.clone();
        let delete_title = chat.title.clone();
        let count = chat.messages.len();
        let subtitle = match &chat.model_name {
            Some(model) => format!("{model} · {count} messages"),
            None => format!("{count} messages"),
        };

        div()
            .id(SharedString::from(chat.id.clone()))
            .flex()
            .items_center()
            .justify_between()
            .gap_3()
            .h(px(56.))
            .px_3()
            .rounded_lg()
            .cursor(CursorStyle::PointingHand)
            .hover(move |s| s.bg(hover_bg))
            .on_click(cx.listener(move |_this, _, _, cx| {
                cx.emit(ChatsEvent::Open(open_id.clone()));
            }))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.text)
                            .font_weight(FontWeight::MEDIUM)
                            .child(chat.title.clone()),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme.text_muted)
                            .child(subtitle),
                    ),
            )
            .child(
                IconButton::new(SharedString::from(format!("del-{}", chat.id)), Icon::Trash)
                    .color(theme.text_muted)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.confirm_delete = Some((delete_id.clone(), delete_title.clone()));
                        cx.notify();
                    })),
            )
    }
}

impl Render for ChatsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();

        let mut list = div().flex().flex_col().gap_1().pb_6();
        if self.chats.is_empty() {
            list = list.child(
                div()
                    .py_8()
                    .text_color(theme.text_muted)
                    .child("No saved chats yet."),
            );
        } else {
            // Clone ids/data already inside row(); iterate by reference.
            let chats = std::mem::take(&mut self.chats);
            for chat in &chats {
                list = list.child(self.row(cx, chat));
            }
            self.chats = chats;
        }

        let modal = self.confirm_delete.clone().map(|(id, title)| {
            ConfirmModal::new("Delete chat", format!("Delete \"{title}\"? This can't be undone."))
                .confirm_label("Delete")
                .danger(true)
                .on_confirm(cx.listener(move |this, _, _, cx| {
                    persistence::delete_chat(&id);
                    this.confirm_delete = None;
                    this.reload(cx);
                }))
                .on_cancel(cx.listener(|this, _, _, cx| {
                    this.confirm_delete = None;
                    cx.notify();
                }))
        });

        div()
            .size_full()
            .relative()
            .flex()
            .flex_col()
            .items_center()
            .child(
                div()
                    .w_full()
                    .max_w(px(CONTENT_MAX_WIDTH))
                    .h_full()
                    .min_h_0()
                    .flex()
                    .flex_col()
                    .px_6()
                    .child(
                        div()
                            .pt_10()
                            .pb_2()
                            .text_xl()
                            .font_weight(FontWeight::MEDIUM)
                            .child("Chat history"),
                    )
                    .child(
                        div()
                            .id("chats-list")
                            .flex_1()
                            .min_h_0()
                            .overflow_y_scroll()
                            .child(list),
                    ),
            )
            .children(modal)
    }
}
