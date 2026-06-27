//! Chat history: saved chats, search, bulk delete, rename in selection mode.

use std::collections::HashSet;

use gpui::{
    AnyElement, Context, CursorStyle, Entity, EventEmitter, FontWeight, IntoElement, Render,
    SharedString, Window, div, prelude::*, px, uniform_list,
};

use super::{
    event::ChatsEvent,
    util::{relative_time, validate_rename_name},
};
use crate::{
    components::{
        Button, ButtonKind, ButtonSize, ConfirmModal, Icon, IconButton, IconEl, InputEvent,
        TextInput,
    },
    persistence::{self, StoredChat},
    theme::{ActiveTheme, layout::CONTENT_MAX_WIDTH},
};

pub struct ChatsView {
    chats: Vec<StoredChat>,
    search: Entity<TextInput>,
    instructions: Entity<TextInput>,
    instructions_open: bool,
    rename_input: Entity<TextInput>,
    rename_open: bool,
    rename_error: Option<&'static str>,
    confirm_delete: Option<(String, String)>,
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
            instructions_open: false,
            rename_input,
            rename_open: false,
            rename_error: None,
            confirm_delete: None,
            selection_mode: false,
            selected: HashSet::new(),
            confirm_bulk_delete: false,
        }
    }

    #[cfg(test)]
    pub fn open_instructions(&mut self, cx: &mut Context<Self>) {
        self.instructions_open = true;
        cx.notify();
    }

    fn open_rename(&mut self, title: &str, cx: &mut Context<Self>) {
        self.rename_open = true;
        self.rename_error = None;
        self.rename_input.update(cx, |input, cx| input.set_text(title, cx));
        cx.notify();
    }

    fn close_rename(&mut self, cx: &mut Context<Self>) {
        self.rename_open = false;
        self.rename_error = None;
        cx.notify();
    }

    fn confirm_rename(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.selected.iter().next().cloned().filter(|_| self.selected.len() == 1)
        else {
            return;
        };
        let text = self.rename_input.read(cx).text();
        match validate_rename_name(&text) {
            Err(msg) => {
                self.rename_error = Some(msg);
                cx.notify();
            }
            Ok(title) => {
                if persistence::rename_chat(&id, &title) {
                    self.close_rename(cx);
                    self.reload(cx);
                } else {
                    self.rename_error = Some("failed to rename chat");
                    cx.notify();
                }
            }
        }
    }

    fn rename_modal(&self, cx: &mut Context<Self>) -> Option<impl IntoElement> {
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
                .bg(gpui::black().opacity(0.5))
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
                        .child(
                            div()
                                .text_color(theme.text)
                                .font_weight(FontWeight::MEDIUM)
                                .child("Rename chat"),
                        )
                        .child(self.rename_input.clone())
                        .children(error.map(|msg| {
                            div().text_xs().text_color(theme.error).child(msg)
                        }))
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
                        IconEl::new(Icon::Plus, theme.text)
                            .size(crate::tokens::icon::MD)
                            // The `+` rotates 45° into an `×` when expanded.
                            .rotate(if open { 45. } else { 0. }),
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
        let subtitle = relative_time(chat.updated_at);

        // Selection checkbox (built from primitives — no Checkbox component yet).
        let mut checkbox = div()
            .size(px(18.))
            .flex_none()
            .rounded(crate::tokens::radius::SM)
            .border_1()
            .border_color(theme.border)
            .flex()
            .items_center()
            .justify_center();
        if selected {
            checkbox =
                checkbox.bg(theme.info).child(IconEl::new(Icon::Check, theme.card).size(crate::tokens::icon::XS));
        }

        // Leading glyph: checkbox in select mode, otherwise the chat icon.
        let mut left = div().flex().items_center().gap_3();
        if selection_mode {
            left = left.child(checkbox);
        } else {
            left = left.child(IconEl::new(Icon::Chats, theme.text_muted).size(crate::tokens::icon::LG));
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
                .child(
                    div()
                        .font_family(crate::theme::FONT_MONO)
                        .text_xs()
                        .text_color(theme.text_muted)
                        .child(subtitle),
                ),
        );

        // Bordered card row (mirai-chat ChatCard); accent border when selected.
        let border = if selected { theme.info } else { theme.border };
        div()
            .id(SharedString::from(chat.id.clone()))
            .flex()
            .items_center()
            .justify_between()
            .gap_3()
            .min_h(px(56.))
            .px_3()
            .py_2()
            .rounded_lg()
            .border_1()
            .border_color(border)
            .bg(theme.card)
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
            .child(left)
            .into_any_element()
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
                .rounded(crate::tokens::radius::SM)
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
                    select_all.bg(theme.info).child(IconEl::new(Icon::Check, theme.card).size(crate::tokens::icon::XS));
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
                actions = actions.child(
                    IconButton::new("rename-one", Icon::Rename)
                        .color(theme.text_muted)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.open_rename(&title, cx);
                        })),
                );
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
                .child(
                    IconButton::new("sel-exit", Icon::Close)
                        .color(theme.text_muted)
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.selection_mode = false;
                            this.selected.clear();
                            cx.notify();
                        })),
                );

            div()
                .flex()
                .items_center()
                .justify_between()
                .mb_3()
                .child(
                    div().flex().items_center().gap_3().child(select_all).child(
                        div().text_sm().text_color(theme.text_muted).child(if count == 0 {
                            "Select all".to_string()
                        } else if all {
                            "All selected".to_string()
                        } else {
                            format!("{count} selected")
                        }),
                    ),
                )
                .child(actions)
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
                        .child(IconEl::new(Icon::Search, theme.text_muted).size(crate::tokens::icon::SM))
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

        let chats_empty = self.chats.is_empty();
        let hit = |chat: &StoredChat| {
            query.is_empty()
                || chat.title.to_lowercase().contains(query)
                || chat.model_name.as_deref().is_some_and(|m| m.to_lowercase().contains(query))
        };
        let filtered: Vec<String> =
            self.chats.iter().filter(|c| hit(c)).map(|c| c.id.clone()).collect();
        let toolbar = self.toolbar(cx, filtered.clone(), chats_empty);

        // Virtualized chat list: only visible rows are built (the saved-chat
        // count grows unbounded). Empty/no-match states render in their place.
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
                    cx.processor(move |this, range: std::ops::Range<usize>, _window, cx| {
                        range
                            .filter_map(|ix| {
                                let id = ids.get(ix)?;
                                let chat = this.chats.iter().find(|c| &c.id == id)?;
                                let selected = this.selected.contains(id);
                                Some(
                                    this.row(cx, chat, this.selection_mode, selected)
                                        .into_any_element(),
                                )
                            })
                            .collect()
                    }),
                )
                .flex_1()
                .min_h_0()
                .into_any_element()
            }
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
                    .child(div().h_px().w_full().bg(theme.border).mb_3())
                    .child(toolbar)
                    .child(list_area),
            )
            .children(modal)
            .children(bulk_modal)
            .children(self.rename_modal(cx))
    }
}

