use std::{collections::HashSet, ops::Range};

use gpui::{Context, Entity, EventEmitter, FontWeight, IntoElement, Render, Window, div, prelude::*, px, uniform_list};

use super::event::ChatsEvent;
use crate::{
    components::{ConfirmModal, InputEvent, TextInput},
    instructions_card::InstructionsCard,
    persistence::{self, StoredChat},
    theme::{ActiveTheme, layout::CONTENT_MAX_WIDTH},
};

pub struct ChatsView {
    pub(super) chats: Vec<StoredChat>,
    pub(super) search: Entity<TextInput>,
    pub(super) instructions: Entity<InstructionsCard>,
    pub(super) rename_input: Entity<TextInput>,
    pub(super) rename_open: bool,
    pub(super) rename_error: Option<&'static str>,
    pub(super) confirm_delete: Option<(String, String)>,
    pub(super) selection_mode: bool,
    pub(super) selected: HashSet<String>,
    pub(super) confirm_bulk_delete: bool,
}

impl EventEmitter<ChatsEvent> for ChatsView {}

impl ChatsView {
    pub fn new(cx: &mut Context<Self>) -> Self {
        let search = cx.new(|cx| TextInput::new(cx, "Search chats…"));
        cx.observe(&search, |_, _, cx| cx.notify()).detach();

        let instructions = cx.new(|cx| InstructionsCard::new(cx, "instr-card", "Tailor the way the model responds"));

        let rename_input = cx.new(|cx| TextInput::new(cx, "Enter chat name"));
        cx.subscribe(&rename_input, |this, _, event, cx| {
            if matches!(event, InputEvent::Submit(_)) {
                this.confirm_rename(cx);
            }
        })
        .detach();

        Self {
            chats: persistence::list_chats(),
            search,
            instructions,
            rename_input,
            rename_open: false,
            rename_error: None,
            confirm_delete: None,
            selection_mode: false,
            selected: HashSet::new(),
            confirm_bulk_delete: false,
        }
    }

    pub fn reload(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        self.chats = persistence::list_chats();

        cx.emit(ChatsEvent::Changed);
        cx.notify();
    }
}

impl Render for ChatsView {
    fn render(
        &mut self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();

        let query = self.search.read(cx).text().to_lowercase();
        let query = query.trim();

        let chats_empty = self.chats.is_empty();
        let hit = |chat: &StoredChat| {
            query.is_empty()
                || chat.title.to_lowercase().contains(query)
                || chat.model_name.as_deref().is_some_and(|m| m.to_lowercase().contains(query))
        };
        let filtered: Vec<String> = self.chats.iter().filter(|c| hit(c)).map(|c| c.id.clone()).collect();
        let toolbar = self.toolbar(cx, filtered.clone(), chats_empty);

        let empty_msg = if chats_empty {
            Some("No chats yet. Start a new conversation!")
        } else if filtered.is_empty() {
            Some("No chats found matching your search.")
        } else {
            None
        };
        let list_area = match empty_msg {
            Some(msg) => div()
                .id("chats-list")
                .flex_1()
                .min_h_0()
                .overflow_y_scroll()
                .child(div().py_8().text_color(theme.text_muted).child(msg))
                .into_any_element(),
            None => {
                let ids = filtered;
                uniform_list(
                    "chats-list",
                    ids.len(),
                    cx.processor(move |this, range: Range<usize>, _window, cx| {
                        range
                            .filter_map(|ix| {
                                let id = ids.get(ix)?;
                                let chat = this.chats.iter().find(|c| &c.id == id)?;
                                let selected = this.selected.contains(id);
                                Some(this.row(cx, chat, this.selection_mode, selected).into_any_element())
                            })
                            .collect()
                    }),
                )
                .flex_1()
                .min_h_0()
                .into_any_element()
            },
        };

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

        let bulk_modal = self.confirm_bulk_delete.then(|| {
            let n = self.selected.len();
            ConfirmModal::new("Delete chats", format!("Delete {n} selected chat(s)? This can't be undone."))
                .confirm_label("Delete")
                .danger(true)
                .on_confirm(cx.listener(|this, _, _, cx| {
                    for id in this.selected.drain() {
                        persistence::delete_chat(&id);
                    }
                    this.confirm_bulk_delete = false;
                    this.selection_mode = false;
                    this.reload(cx);
                }))
                .on_cancel(cx.listener(|this, _, _, cx| {
                    this.confirm_bulk_delete = false;
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
                    .child(div().pt_10().pb_2().text_xl().font_weight(FontWeight::MEDIUM).child("Chat history"))
                    .child(div().mb_3().child(self.instructions.clone()))
                    .child(div().h_px().w_full().bg(theme.border).mb_3())
                    .child(toolbar)
                    .child(list_area),
            )
            .children(modal)
            .children(bulk_modal)
            .children(self.rename_modal(cx))
    }
}
