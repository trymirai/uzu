//! Local Models — two-level browser matching mirai-chat: a family list, then a
//! family detail (Installed/Available, download/pause/resume/delete; tapping an
//! installed model starts a chat).

use gpui::{
    Context, CursorStyle, Entity, EventEmitter, FontWeight, IntoElement, Render, SharedString,
    Window, div, prelude::*, px,
};
use uzu::{storage::types::DownloadPhase, types::model::Model};

use crate::{
    components::{ConfirmModal, Icon, IconButton, IconEl, Loader, TextInput, VendorIcon},
    models_store::ModelsStore,
    theme::{ActiveTheme, layout::CONTENT_MAX_WIDTH},
};

/// Emitted when the user picks an installed model to chat with.
pub enum LocalModelsEvent {
    UseModel(Model),
}

struct ModelVm {
    id: String,
    name: String,
    size: String,
    quant: String,
    phase: DownloadPhase,
    progress: f32,
    is_mirai: bool,
}

impl ModelVm {
    fn installed(&self) -> bool {
        matches!(self.phase, DownloadPhase::Downloaded {})
    }
    fn downloading(&self) -> bool {
        matches!(self.phase, DownloadPhase::Downloading {})
    }
    fn paused(&self) -> bool {
        matches!(self.phase, DownloadPhase::Paused {})
    }
}

struct FamilyVm {
    key: String,
    name: String,
    vendor: String,
    range: Option<String>,
    has_mirai: bool,
    models: Vec<ModelVm>,
}

impl FamilyVm {
    fn installed_count(&self) -> usize {
        self.models.iter().filter(|m| m.installed()).count()
    }
}

pub struct LocalModelsView {
    store: Entity<ModelsStore>,
    search: Entity<TextInput>,
    /// Family identifier being viewed in detail (None = family list).
    selected_family: Option<String>,
    /// (id, name) pending delete confirmation.
    confirm_delete: Option<(String, String)>,
}

impl EventEmitter<LocalModelsEvent> for LocalModelsView {}

impl LocalModelsView {
    pub fn new(store: Entity<ModelsStore>, cx: &mut Context<Self>) -> Self {
        let search = cx.new(|cx| TextInput::new(cx, "Search families"));
        cx.observe(&store, |_, _, cx| cx.notify()).detach();
        cx.observe(&search, |_, _, cx| cx.notify()).detach();
        Self {
            store,
            search,
            selected_family: None,
            confirm_delete: None,
        }
    }

    fn open_family(&mut self, key: String, cx: &mut Context<Self>) {
        self.selected_family = Some(key);
        self.search.update(cx, |i, cx| i.clear(cx));
        cx.notify();
    }

    fn back_to_families(&mut self, cx: &mut Context<Self>) {
        self.selected_family = None;
        self.search.update(cx, |i, cx| i.clear(cx));
        cx.notify();
    }

    /// Group the store's models into families, preserving first-seen order.
    fn families(&self, cx: &Context<Self>) -> Vec<FamilyVm> {
        let store = self.store.read(cx);
        let mut order: Vec<String> = Vec::new();
        let mut families: std::collections::HashMap<String, FamilyVm> = Default::default();

        for row in &store.rows {
            let (key, name, vendor) = match &row.model.family {
                Some(f) => (f.identifier.clone(), f.name(), f.vendor.name()),
                None => ("other".to_string(), "Other".to_string(), String::new()),
            };
            let quant = quant_label(&row.model);
            let is_mirai = row
                .model
                .quantization
                .as_ref()
                .map(|q| {
                    q.method.to_lowercase().contains("mirai")
                        || q.identifier.to_lowercase().contains("mirai")
                })
                .unwrap_or(false);
            // Prefer the actual download size (total_bytes) when known, else the
            // registry's nominal properties size.
            let bytes = row
                .state
                .as_ref()
                .map(|s| s.total_bytes)
                .filter(|b| *b > 0)
                .unwrap_or_else(|| row.size_bytes());
            let vm = ModelVm {
                id: row.id().to_string(),
                name: row.name(),
                size: format_size(bytes),
                quant,
                phase: row.phase(),
                progress: row.progress(),
                is_mirai,
            };
            let entry = families.entry(key.clone()).or_insert_with(|| {
                order.push(key.clone());
                FamilyVm {
                    key: key.clone(),
                    name,
                    vendor,
                    range: None,
                    has_mirai: false,
                    models: Vec::new(),
                }
            });
            entry.has_mirai = entry.has_mirai || is_mirai;
            entry.models.push(vm);
        }

        // Compute parameter-count range per family (parsed from model names).
        for fam in families.values_mut() {
            let mut params: Vec<f64> = fam
                .models
                .iter()
                .filter_map(|m| parse_params(&m.name))
                .collect();
            params.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if let (Some(min), Some(max)) = (params.first(), params.last()) {
                fam.range = Some(if (min - max).abs() < f64::EPSILON {
                    format_params(*max)
                } else {
                    format!("{} – {}", format_params(*min), format_params(*max))
                });
            }
        }

        order
            .into_iter()
            .filter_map(|k| families.remove(&k))
            .collect()
    }

    fn chip(&self, text: String, accent: bool, cx: &Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let (fg, bg) = if accent {
            (theme.success, theme.bg_sub)
        } else {
            (theme.text_muted, theme.bg_sub)
        };
        div()
            .flex()
            .items_center()
            .h(px(24.))
            .px_2()
            .rounded_md()
            .bg(bg)
            .text_color(fg)
            .text_xs()
            .child(text)
    }

    fn family_row(&self, cx: &mut Context<Self>, fam: &FamilyVm) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let key = fam.key.clone();
        let installed = fam.installed_count();
        let total = fam.models.len();

        let mut chips = div().flex().items_center().gap_2();
        if fam.has_mirai {
            chips = chips.child(self.chip("Mirai quantizations".to_string(), true, cx));
        }
        chips = chips.child(self.chip(
            format!("{total} model{}", if total == 1 { "" } else { "s" }),
            false,
            cx,
        ));
        if installed > 0 {
            chips = chips.child(self.chip(
                format!("{installed} installed model{}", if installed == 1 { "" } else { "s" }),
                true,
                cx,
            ));
        }
        if let Some(range) = &fam.range {
            chips = chips.child(self.chip(range.clone(), false, cx));
        }

        div()
            .id(SharedString::from(format!("fam-{}", fam.key)))
            .flex()
            .items_center()
            .gap_3()
            .h(px(56.))
            .px_4()
            .rounded_lg()
            .cursor(CursorStyle::PointingHand)
            .hover(move |s| s.bg(hover))
            .on_click(cx.listener(move |this, _, _, cx| this.open_family(key.clone(), cx)))
            .child(VendorIcon::new(fam.vendor.clone()).size(20.))
            .child(
                div()
                    .flex_1()
                    .flex()
                    .items_baseline()
                    .gap_2()
                    .child(
                        div()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(theme.text)
                            .child(fam.name.clone()),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.text_muted)
                            .child(format!("from {}", fam.vendor)),
                    ),
            )
            .child(chips)
            .child(IconEl::new(Icon::ChevronRight, theme.text_muted).size(16.))
    }

    fn model_row(&self, cx: &mut Context<Self>, vm: &ModelVm) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let id = vm.id.clone();

        // Left info area — clickable to start a chat when installed.
        let name_label = div()
            .text_sm()
            .text_color(theme.text)
            .child(vm.name.clone());
        let info = if vm.installed() {
            let chat_id = id.clone();
            div()
                .id(SharedString::from(format!("use-{}", vm.id)))
                .flex_1()
                .flex()
                .items_center()
                .gap_2()
                .h_full()
                .cursor(CursorStyle::PointingHand)
                .on_click(cx.listener(move |this, _, _, cx| {
                    if let Some(model) = this
                        .store
                        .read(cx)
                        .rows
                        .iter()
                        .find(|r| r.id() == chat_id)
                        .map(|r| r.model.clone())
                    {
                        cx.emit(LocalModelsEvent::UseModel(model));
                    }
                }))
                .child(IconEl::new(Icon::Models, theme.text_muted).size(18.))
                .child(name_label)
                .into_any_element()
        } else {
            div()
                .flex_1()
                .flex()
                .items_center()
                .gap_2()
                .h_full()
                .child(IconEl::new(Icon::Models, theme.text_muted).size(18.))
                .child(name_label)
                .into_any_element()
        };

        // Right action area (varies by phase).
        let action = if vm.installed() {
            let del_id = id.clone();
            let del_name = vm.name.clone();
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(IconEl::new(Icon::Check, theme.success).size(15.))
                .child(
                    IconButton::new(SharedString::from(format!("del-{}", vm.id)), Icon::Trash)
                        .color(theme.text_muted)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.confirm_delete = Some((del_id.clone(), del_name.clone()));
                            cx.notify();
                        })),
                )
                .into_any_element()
        } else if vm.downloading() || vm.paused() {
            let toggle_id = id.clone();
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(
                    div()
                        .text_xs()
                        .text_color(theme.text_muted)
                        .child(format!("{:.0}%", vm.progress * 100.0)),
                )
                .child(
                    IconButton::new(
                        SharedString::from(format!("tog-{}", vm.id)),
                        if vm.paused() { Icon::Download } else { Icon::Pause },
                    )
                    .color(theme.text_muted)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        let id = toggle_id.clone();
                        this.store.update(cx, |s, cx| s.toggle_download(id, cx));
                    })),
                )
                .into_any_element()
        } else {
            let dl_id = id.clone();
            IconButton::new(SharedString::from(format!("dl-{}", vm.id)), Icon::Download)
                .color(theme.text_muted)
                .on_click(cx.listener(move |this, _, _, cx| {
                    let id = dl_id.clone();
                    this.store.update(cx, |s, cx| s.toggle_download(id, cx));
                }))
                .into_any_element()
        };

        div()
            .flex()
            .items_center()
            .gap_4()
            .h(px(52.))
            .px_4()
            .rounded_lg()
            .when(vm.is_mirai, |el| {
                el.border_l_2().border_color(theme.success)
            })
            .hover(move |s| s.bg(hover))
            .child(info)
            .child(
                div()
                    .w(px(80.))
                    .text_sm()
                    .text_color(theme.text_muted)
                    .child(vm.size.clone()),
            )
            .child(
                div()
                    .w(px(140.))
                    .text_sm()
                    .text_color(theme.text_muted)
                    .child(vm.quant.clone()),
            )
            .child(action)
    }
}

impl Render for LocalModelsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let query = self.search.read(cx).text().to_lowercase();
        let families = self.families(cx);
        let selected = self.selected_family.clone();

        // Delete-confirm modal (shared across both levels).
        let modal = self.confirm_delete.clone().map(|(id, name)| {
            ConfirmModal::new(
                "Delete model",
                format!("Delete \"{name}\"? You can download it again later."),
            )
            .confirm_label("Delete")
            .danger(true)
            .on_confirm(cx.listener(move |this, _, _, cx| {
                this.store.update(cx, |s, cx| s.delete(id.clone(), cx));
                this.confirm_delete = None;
                cx.notify();
            }))
            .on_cancel(cx.listener(|this, _, _, cx| {
                this.confirm_delete = None;
                cx.notify();
            }))
        });

        // Header (title or back button) + search box.
        let header = match &selected {
            Some(key) => {
                let title = families
                    .iter()
                    .find(|f| &f.key == key)
                    .map(|f| (f.name.clone(), f.vendor.clone()))
                    .unwrap_or_else(|| ("Models".to_string(), String::new()));
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        IconButton::new("models-back", Icon::ChevronLeft)
                            .color(theme.text_muted)
                            .on_click(cx.listener(|this, _, _, cx| this.back_to_families(cx))),
                    )
                    .child(VendorIcon::new(title.1.clone()).size(22.))
                    .child(
                        div()
                            .text_xl()
                            .font_weight(FontWeight::MEDIUM)
                            .child(title.0),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.text_muted)
                            .child(format!("from {}", title.1)),
                    )
                    .into_any_element()
            }
            None => div()
                .flex()
                .items_center()
                .gap_2()
                .child(IconEl::new(Icon::Models, theme.text).size(22.))
                .child(
                    div()
                        .text_xl()
                        .font_weight(FontWeight::MEDIUM)
                        .child("Choose local model to chat"),
                )
                .into_any_element(),
        };

        // Body list.
        let mut list = div().flex().flex_col().gap_1().pb_6();
        match &selected {
            None => {
                let visible: Vec<&FamilyVm> = families
                    .iter()
                    .filter(|f| {
                        query.is_empty()
                            || f.name.to_lowercase().contains(&query)
                            || f.vendor.to_lowercase().contains(&query)
                    })
                    .collect();
                if visible.is_empty() {
                    list = list.child(self.empty_message(cx));
                } else {
                    for fam in visible {
                        list = list.child(self.family_row(cx, fam));
                    }
                }
            }
            Some(key) => {
                if let Some(fam) = families.iter().find(|f| &f.key == key) {
                    let matched: Vec<&ModelVm> = fam
                        .models
                        .iter()
                        .filter(|m| query.is_empty() || m.name.to_lowercase().contains(&query))
                        .collect();
                    let installed: Vec<&ModelVm> =
                        matched.iter().copied().filter(|m| m.installed()).collect();
                    let available: Vec<&ModelVm> =
                        matched.iter().copied().filter(|m| !m.installed()).collect();

                    if !installed.is_empty() {
                        list = list.child(section_header("Installed models", &theme));
                        for vm in installed {
                            list = list.child(self.model_row(cx, vm));
                        }
                    }
                    if !available.is_empty() {
                        list = list.child(section_header("Available models", &theme));
                        for vm in available {
                            list = list.child(self.model_row(cx, vm));
                        }
                    }
                }
            }
        }

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
                            .pb_3()
                            .flex()
                            .items_center()
                            .justify_between()
                            .gap_4()
                            .child(header)
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .w(px(280.))
                                    .h(px(32.))
                                    .px_3()
                                    .rounded_lg()
                                    .border_1()
                                    .border_color(theme.border)
                                    .bg(theme.card)
                                    .child(IconEl::new(Icon::Search, theme.text_muted).size(15.))
                                    .child(div().flex_1().child(self.search.clone())),
                            ),
                    )
                    .child(
                        div()
                            .id("models-list")
                            .flex_1()
                            .min_h_0()
                            .overflow_y_scroll()
                            .child(list),
                    ),
            )
            .children(modal)
    }
}

impl LocalModelsView {
    fn empty_message(&self, cx: &Context<Self>) -> gpui::AnyElement {
        let theme = cx.theme().clone();
        let store = self.store.read(cx);
        if store.loading {
            return div()
                .py_8()
                .flex()
                .justify_center()
                .child(Loader::new().label("Loading models…"))
                .into_any_element();
        }
        let msg = match &store.error {
            Some(err) => format!("Failed to load models: {err}"),
            None => "No models".to_string(),
        };
        div().py_8().text_color(theme.text_muted).child(msg).into_any_element()
    }
}

fn section_header(label: &str, theme: &crate::theme::Theme) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .gap_4()
        .pt_4()
        .pb_1()
        .px_4()
        .text_xs()
        .font_weight(FontWeight::MEDIUM)
        .text_color(theme.text_muted)
        .child(div().flex_1().child(label.to_uppercase()))
        .child(div().w(px(80.)).child("SIZE"))
        .child(div().w(px(140.)).child("QUANTIZATION"))
        .child(div().w(px(56.)))
}

fn quant_label(model: &Model) -> String {
    // `method` is already the clean short label, e.g. "mirai-m", "mirai-l",
    // "mlx" → "MIRAI-M · 4-bit", "MLX · 4-bit".
    match &model.quantization {
        Some(q) => format!("{} · {}-bit", q.method.to_uppercase(), q.bits_per_weight),
        None => "Unquantized".to_string(),
    }
}

/// Parse a parameter count (in millions) from a model name token like `0.8B`,
/// `4B`, `27B`, `800M`.
fn parse_params(name: &str) -> Option<f64> {
    for raw in name.split(|c: char| c == ' ' || c == '-') {
        let token = raw.trim();
        if let Some(num) = token
            .strip_suffix('B')
            .or_else(|| token.strip_suffix('b'))
        {
            if let Ok(v) = num.parse::<f64>() {
                return Some(v * 1000.0);
            }
        }
        if let Some(num) = token
            .strip_suffix('M')
            .or_else(|| token.strip_suffix('m'))
        {
            if let Ok(v) = num.parse::<f64>() {
                return Some(v);
            }
        }
    }
    None
}

fn format_params(millions: f64) -> String {
    if millions >= 1000.0 {
        let b = millions / 1000.0;
        if (b.fract()).abs() < f64::EPSILON {
            format!("{b:.0}B")
        } else {
            format!("{b:.1}B")
        }
    } else {
        format!("{millions:.0}M")
    }
}

fn format_size(bytes: i64) -> String {
    if bytes <= 0 {
        return "—".to_string();
    }
    let b = bytes as f64;
    const GB: f64 = 1_000_000_000.0;
    const MB: f64 = 1_000_000.0;
    if b >= GB {
        format!("{:.1} GB", b / GB)
    } else {
        format!("{:.0} MB", b / MB)
    }
}
