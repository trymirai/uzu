//! Chat history screen: lists saved chats (newest first), open on click, delete
//! per row.

use std::collections::HashSet;

use gpui::{
    AnyElement, Context, CursorStyle, Entity, EventEmitter, FontWeight, IntoElement, Render,
    SharedString, Window, div, prelude::*, px,
};

use crate::{
    components::{
        Button, ButtonKind, ButtonSize, ConfirmModal, Icon, IconButton, IconEl, InputEvent,
        TextInput,
    },
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
    /// Multi-select state ("Select" mode for bulk delete).
    selection_mode: bool,
    selected: HashSet<String>,
    confirm_bulk_delete: bool,
}

impl EventEmitter<ChatsEvent> for ChatsView {}

impl ChatsView {
    pub fn new(cx: &mut Context<Self>) -> Self {
        let search = cx.new(|cx| TextInput::new(cx, "Search chats…"));
        cx.observe(&search, |_, _, cx| cx.notify()).detach();

        let instructions =
            cx.new(|cx| TextInput::new(cx, "Tailor the way the model responds").multiline(false, 3, 6));
        let current = persistence::global_instructions();
        if !current.is_empty() {
            instructions.update(cx, |input, cx| input.set_text(current, cx));
        }
        cx.subscribe(&instructions, |_this, _input, event, _cx| match event {
            InputEvent::Submit(text) | InputEvent::Changed(text) => {
                persistence::set_global_instructions(text)
            }
        })
        .detach();

        Self {
            chats: persistence::list_chats(),
            search,
            instructions,
            instructions_open: false,
            confirm_delete: None,
            selection_mode: false,
            selected: HashSet::new(),
            confirm_bulk_delete: false,
        }
    }

    /// Expandable "Add instructions to all chats" card (mirai-chat parity). The
    /// editor persists to the global-instructions store on Enter.
    fn instructions_card(&self, cx: &mut Context<Self>) -> AnyElement {
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
        card.into_any_element()
    }

    pub fn reload(&mut self, cx: &mut Context<Self>) {
        self.chats = persistence::list_chats();
        cx.notify();
    }

    fn row(
        &self,
        cx: &mut Context<Self>,
        chat: &StoredChat,
        selection_mode: bool,
        selected: bool,
    ) -> AnyElement {
        let theme = cx.theme().clone();
        let hover_bg = theme.bg_hover;
        let click_id = chat.id.clone();
        let delete_id = chat.id.clone();
        let delete_title = chat.title.clone();
        let count = chat.messages.len();
        let subtitle = match &chat.model_name {
            Some(model) => format!("{model} · {count} messages"),
            None => format!("{count} messages"),
        };

        // Selection checkbox (built from primitives — no Checkbox component yet).
        let mut checkbox = div()
            .size(px(18.))
            .flex_none()
            .rounded(px(4.))
            .border_1()
            .border_color(theme.border)
            .flex()
            .items_center()
            .justify_center();
        if selected {
            checkbox =
                checkbox.bg(theme.info).child(IconEl::new(Icon::Check, theme.card).size(12.));
        }

        let mut left = div().flex().items_center().gap_3();
        if selection_mode {
            left = left.child(checkbox);
        }
        left = left.child(
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
                .child(div().text_xs().text_color(theme.text_muted).child(subtitle)),
        );

        let mut row = div()
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
            .on_click(cx.listener(move |this, _, _, cx| {
                if this.selection_mode {
                    if !this.selected.remove(&click_id) {
                        this.selected.insert(click_id.clone());
                    }
                    cx.notify();
                } else {
                    cx.emit(ChatsEvent::Open(click_id.clone()));
                }
            }))
            .child(left);

        if !selection_mode {
            row = row.child(
                IconButton::new(SharedString::from(format!("del-{}", chat.id)), Icon::Trash)
                    .color(theme.text_muted)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        // Don't let the click bubble to the row (which would open
                        // the chat); just raise the delete-confirm modal.
                        cx.stop_propagation();
                        this.confirm_delete = Some((delete_id.clone(), delete_title.clone()));
                        cx.notify();
                    })),
            );
        }
        row.into_any_element()
    }

    /// "Your chats" row (search + Select), or the selection toolbar in select
    /// mode (select-all · count · Delete · exit), mirroring mirai-chat.
    fn toolbar(&self, cx: &mut Context<Self>, filtered: Vec<String>, empty: bool) -> AnyElement {
        let theme = cx.theme().clone();

        if self.selection_mode {
            let total = filtered.len();
            let count = self.selected.len();
            let all = total > 0 && count == total;

            let mut select_all = div()
                .id("select-all")
                .size(px(18.))
                .flex_none()
                .rounded(px(4.))
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
                    select_all.bg(theme.info).child(IconEl::new(Icon::Check, theme.card).size(12.));
            }

            div()
                .flex()
                .items_center()
                .justify_between()
                .mb_3()
                .child(
                    div().flex().items_center().gap_3().child(select_all).child(
                        div()
                            .text_sm()
                            .text_color(theme.text_muted)
                            .child(format!("{count} selected")),
                    ),
                )
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap_2()
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
                        .child(
                            IconButton::new("sel-exit", Icon::Close)
                                .color(theme.text_muted)
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.selection_mode = false;
                                    this.selected.clear();
                                    cx.notify();
                                })),
                        ),
                )
                .into_any_element()
        } else {
            div()
                .flex()
                .items_center()
                .gap_3()
                .mb_3()
                .child(
                    div()
                        .text_sm()
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(theme.text)
                        .child("Your chats"),
                )
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
                        .child(IconEl::new(Icon::Search, theme.text_muted).size(14.))
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

impl Render for ChatsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();

        let query = self.search.read(cx).text().to_lowercase();
        let query = query.trim();

        let chats = std::mem::take(&mut self.chats);
        let chats_empty = chats.is_empty();
        let hit = |chat: &StoredChat| {
            query.is_empty()
                || chat.title.to_lowercase().contains(query)
                || chat.model_name.as_deref().is_some_and(|m| m.to_lowercase().contains(query))
        };
        let filtered: Vec<String> = chats.iter().filter(|c| hit(c)).map(|c| c.id.clone()).collect();
        let toolbar = self.toolbar(cx, filtered, chats_empty);

        let mut list = div().flex().flex_col().gap_1().pb_6();
        if chats_empty {
            list = list.child(
                div().py_8().text_color(theme.text_muted).child("No saved chats yet."),
            );
        } else {
            let mut shown = 0usize;
            for chat in &chats {
                if hit(chat) {
                    let selected = self.selected.contains(&chat.id);
                    list = list.child(self.row(cx, chat, self.selection_mode, selected));
                    shown += 1;
                }
            }
            if shown == 0 {
                list = list.child(
                    div().py_8().text_color(theme.text_muted).child("No chats match your search."),
                );
            }
        }
        self.chats = chats;

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
            ConfirmModal::new(
                "Delete chats",
                format!("Delete {n} selected chat(s)? This can't be undone."),
            )
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
                    .child(
                        div()
                            .pt_10()
                            .pb_2()
                            .text_xl()
                            .font_weight(FontWeight::MEDIUM)
                            .child("Chat history"),
                    )
                    .child(self.instructions_card(cx))
                    .child(toolbar)
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
            .children(bulk_modal)
    }
}
