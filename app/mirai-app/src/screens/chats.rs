//! Chat history screen: lists saved chats (newest first), open on click, delete
//! per row.

use gpui::{
    Context, CursorStyle, Entity, EventEmitter, FontWeight, IntoElement, Render, SharedString,
    Window, div, prelude::*, px,
};

use crate::{
    components::{ConfirmModal, Icon, IconButton, IconEl, InputEvent, TextInput},
    persistence::{self, StoredChat},
    theme::{ActiveTheme, layout::CONTENT_MAX_WIDTH},
};

/// Emitted to the shell to open a saved chat.
pub enum ChatsEvent {
    Open(String),
}

pub struct ChatsView {
    chats: Vec<StoredChat>,
    search: Entity<TextInput>,
    /// Global-instructions editor behind the expandable card.
    instructions: Entity<TextInput>,
    instructions_open: bool,
    /// (id, title) of a chat pending delete confirmation.
    confirm_delete: Option<(String, String)>,
}

impl EventEmitter<ChatsEvent> for ChatsView {}

impl ChatsView {
    pub fn new(cx: &mut Context<Self>) -> Self {
        let search = cx.new(|cx| TextInput::new(cx, "Search chats…"));
        cx.observe(&search, |_, _, cx| cx.notify()).detach();

        let instructions = cx.new(|cx| TextInput::new(cx, "Tailor the way the model responds"));
        let current = persistence::global_instructions();
        if !current.is_empty() {
            instructions.update(cx, |input, cx| input.set_text(current, cx));
        }
        cx.subscribe(&instructions, |_this, _input, event, _cx| match event {
            InputEvent::Submit(text) => persistence::set_global_instructions(text),
        })
        .detach();

        Self {
            chats: persistence::list_chats(),
            search,
            instructions,
            instructions_open: false,
            confirm_delete: None,
        }
    }

    /// Expandable "Add instructions to all chats" card (mirai-chat parity). The
    /// editor persists to the global-instructions store on Enter.
    fn instructions_card(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let open = self.instructions_open;

        let header = div()
            .id("instr-card")
            .flex()
            .items_center()
            .justify_between()
            .gap_3()
            .px_4()
            .py_3()
            .cursor(CursorStyle::PointingHand)
            .hover(move |s| s.bg(hover))
            .on_click(cx.listener(|this, _, _, cx| {
                this.instructions_open = !this.instructions_open;
                cx.notify();
            }))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        IconEl::new(if open { Icon::Close } else { Icon::Plus }, theme.text)
                            .size(16.),
                    )
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(theme.text)
                            .child("Add instructions to all chats"),
                    ),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(theme.text_muted)
                    .child("Tailor the way the model responds"),
            );

        let mut card = div()
            .w_full()
            .mb_3()
            .rounded_lg()
            .border_1()
            .border_color(theme.border)
            .bg(theme.card)
            .child(header);

        if open {
            card = card.child(
                div().px_4().pb_3().child(
                    div()
                        .w_full()
                        .px_3()
                        .py_2()
                        .rounded_md()
                        .border_1()
                        .border_color(theme.border)
                        .bg(theme.bg)
                        .child(self.instructions.clone()),
                ),
            );
        }
        card
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

        let query = self.search.read(cx).text().to_lowercase();
        let query = query.trim();

        let mut list = div().flex().flex_col().gap_1().pb_6();
        if self.chats.is_empty() {
            list = list.child(
                div().py_8().text_color(theme.text_muted).child("No saved chats yet."),
            );
        } else {
            // row() clones the ids it needs, so iterate by reference.
            let chats = std::mem::take(&mut self.chats);
            let mut shown = 0usize;
            for chat in &chats {
                let hit = query.is_empty()
                    || chat.title.to_lowercase().contains(query)
                    || chat.model_name.as_deref().is_some_and(|m| m.to_lowercase().contains(query));
                if hit {
                    list = list.child(self.row(cx, chat));
                    shown += 1;
                }
            }
            self.chats = chats;
            if shown == 0 {
                list = list.child(
                    div().py_8().text_color(theme.text_muted).child("No chats match your search."),
                );
            }
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
                    .child(self.instructions_card(cx))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .w_full()
                            .mb_3()
                            .px_3()
                            .py_2()
                            .rounded_lg()
                            .border_1()
                            .border_color(theme.border)
                            .bg(theme.card)
                            .child(IconEl::new(Icon::Search, theme.text_muted).size(14.))
                            .child(self.search.clone()),
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
