//! Local model families and family detail (download, chat).

use gpui::{
    Context, CursorStyle, Entity, EventEmitter, FontWeight, IntoElement, Render, SharedString,
    Window, div, prelude::*, px, relative,
};
use uzu::{storage::types::DownloadPhase, types::model::Model};

use crate::{
    components::{ConfirmModal, Icon, IconButton, IconEl, Loader, TextInput, VendorIcon},
    device_info,
    model_recommend,
    model_sort::{self, ModelSort},
    models_store::ModelsStore,
    theme::{ActiveTheme, layout::CONTENT_MAX_WIDTH},
};

/// Emitted when the user picks an installed model to chat with.
pub enum LocalModelsEvent {
    UseModel(Model),
}

#[derive(Clone)]
struct ModelVm {
    id: String,
    name: String,
    size: String,
    bytes: i64,
    quant: String,
    phase: DownloadPhase,
    progress: f32,
    is_mirai: bool,
    recommended: bool,
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
    icon_url: Option<String>,
    range: Option<String>,
    has_mirai: bool,
    last_installed_at: u64,
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
    selected_family: Option<String>,
    confirm_delete: Option<(String, String)>,
    device_label: String,
    recommended_id: Option<String>,
    sort: ModelSort,
    sort_open: bool,
    /// Memoized `families()` output, tagged with the theme it was built for.
    /// Rebuilt only when the store, recommended id, or theme changes — not on
    /// every frame (search/sort are applied to this cached list in `render`).
    families_cache: Option<(bool, Vec<FamilyVm>)>,
}

impl EventEmitter<LocalModelsEvent> for LocalModelsView {}

impl LocalModelsView {
    pub fn new(store: Entity<ModelsStore>, cx: &mut Context<Self>) -> Self {
        let search = cx.new(|cx| TextInput::new(cx, "Search families"));
        // A store change (catalog load, download progress) invalidates the
        // families cache so it rebuilds on the next render; a search change only
        // re-renders (the query is applied to the cached list).
        cx.observe(&store, |this, _, cx| {
            this.families_cache = None;
            cx.notify();
        })
        .detach();
        cx.observe(&search, |_, _, cx| cx.notify()).detach();
        Self::spawn_recommend(cx);
        Self {
            store,
            search,
            selected_family: None,
            confirm_delete: None,
            device_label: device_info::description(),
            recommended_id: None,
            sort: ModelSort::default(),
            sort_open: false,
            families_cache: None,
        }
    }

    fn spawn_recommend(cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| {
            let id = model_recommend::fetch_repo_id().await;
            let _ = this.update(cx, |view, cx| {
                view.recommended_id = id;
                view.families_cache = None;
                cx.notify();
            });
        })
        .detach();
    }

    pub fn open_family(&mut self, key: String, cx: &mut Context<Self>) {
        self.selected_family = Some(key);
        self.search.update(cx, |i, cx| i.clear(cx));
        cx.notify();
    }

    fn back_to_families(&mut self, cx: &mut Context<Self>) {
        self.selected_family = None;
        self.search.update(cx, |i, cx| i.clear(cx));
        cx.notify();
    }

    fn families(&self, cx: &Context<Self>) -> Vec<FamilyVm> {
        let dark = cx.theme().dark;
        let store = self.store.read(cx);
        let recommended_id = self.recommended_id.as_deref();
        let mut order: Vec<String> = Vec::new();
        let mut families: std::collections::HashMap<String, FamilyVm> = Default::default();
        let mut recommended_family: Option<String> = None;

        for row in &store.rows {
            let (key, name, vendor) = match &row.model.family {
                Some(f) => (f.identifier.clone(), f.name(), f.vendor.name()),
                None => ("other".to_string(), "Other".to_string(), String::new()),
            };
            if recommended_id == Some(row.id()) {
                recommended_family = Some(key.clone());
            }
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
            let bytes = row.display_size_bytes();
            let installed_at = store.installed_at(row.id());
            let vm = ModelVm {
                id: row.id().to_string(),
                name: row.name(),
                size: format_size(bytes),
                bytes,
                quant,
                phase: row.phase(),
                progress: row.progress(),
                is_mirai,
                recommended: recommended_id == Some(row.id()),
            };
            let entry = families.entry(key.clone()).or_insert_with(|| {
                order.push(key.clone());
                FamilyVm {
                    key: key.clone(),
                    name,
                    vendor,
                    icon_url: row.icon_url(dark),
                    range: None,
                    has_mirai: false,
                    last_installed_at: 0,
                    models: Vec::new(),
                }
            });
            entry.has_mirai = entry.has_mirai || is_mirai;
            entry.last_installed_at = entry.last_installed_at.max(installed_at);
            entry.models.push(vm);
        }

        for fam in families.values_mut() {
            let mut params: Vec<f64> = fam
                .models
                .iter()
                .filter_map(|m| model_sort::parse_params(&m.name))
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

        let mut list: Vec<FamilyVm> = order
            .into_iter()
            .filter_map(|k| families.remove(&k))
            .collect();

        let rec_key = recommended_family;
        list.sort_by(|a, b| {
            let rank = |f: &FamilyVm| -> u8 {
                if f.has_mirai {
                    0
                } else if rec_key.as_deref() == Some(f.key.as_str()) {
                    1
                } else if f.last_installed_at > 0 {
                    2
                } else {
                    3
                }
            };
            rank(a)
                .cmp(&rank(b))
                .then_with(|| match (rank(a), rank(b)) {
                    (2, 2) => b.last_installed_at.cmp(&a.last_installed_at),
                    _ => a.name.cmp(&b.name),
                })
        });
        list
    }

    /// Rebuild `families_cache` only if it's missing or was built for a
    /// different theme. `render` then takes the cached list, applies the live
    /// search/sort, and restores it.
    fn ensure_families(&mut self, cx: &Context<Self>) {
        let dark = cx.theme().dark;
        let valid = matches!(&self.families_cache, Some((d, _)) if *d == dark);
        if !valid {
            let built = self.families(cx);
            self.families_cache = Some((dark, built));
        }
    }

    fn sort_models(&self, models: &mut [ModelVm]) {
        match self.sort {
            ModelSort::Size => models.sort_by(|a, b| a.bytes.cmp(&b.bytes).then_with(|| model_sort::sort_by_name(&a.name, &b.name))),
            ModelSort::Name => models.sort_by(|a, b| model_sort::sort_by_name(&a.name, &b.name)),
            ModelSort::Newest => models.sort_by(|a, b| model_sort::sort_by_newest(&a.name, &b.name)),
        }
    }

    fn chip(&self, text: String, accent: bool, cx: &Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        // Outlined pill (mirai-chat parity): green text+border for accent chips,
        // a transparent pill with a visible outline + muted text for neutral ones
        // (no grey fill).
        let (fg, bg, border) = if accent {
            (theme.success, theme.success.opacity(0.08), theme.success.opacity(0.45))
        } else {
            (theme.text_muted, gpui::transparent_black(), theme.button_border)
        };
        div()
            .flex()
            .items_center()
            .h(px(26.))
            .px_2p5()
            .rounded_lg()
            .border_1()
            .border_color(border)
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
            // Bordered card row (mirai-chat parity).
            .border_1()
            .border_color(theme.border)
            .bg(theme.card)
            .cursor(CursorStyle::PointingHand)
            .hover(move |s| s.bg(hover))
            .on_click(cx.listener(move |this, _, _, cx| this.open_family(key.clone(), cx)))
            .child(VendorIcon::new(fam.vendor.clone()).size(crate::tokens::icon::XL).icon_url(fam.icon_url.clone()))
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
            .child(IconEl::new(Icon::ChevronRight, theme.text_muted).size(crate::tokens::icon::MD))
    }

    fn model_row(
        &self,
        cx: &mut Context<Self>,
        vm: &ModelVm,
        vendor: &str,
        icon_url: Option<&str>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let id = vm.id.clone();

        // Left info area — clickable to start a chat when installed. Shows the
        // provider logo (same as the family list), not a generic glyph.
        let vendor_icon = VendorIcon::new(vendor.to_string())
            .size(crate::tokens::icon::XL)
            .icon_url(icon_url.map(|u| u.to_string()));
        let name_label = div()
            .flex()
            .items_center()
            .gap_2()
            .child(
                div()
                    .text_sm()
                    .text_color(theme.text)
                    .child(vm.name.clone()),
            )
            .when(vm.recommended, |el| {
                el.child(self.chip("Recommended".to_string(), true, cx))
            });
        let chat_id = id.clone();
        let info = div()
            .id(SharedString::from(format!("use-{}", vm.id)))
            .flex_1()
            .flex()
            .items_center()
            .gap_2()
            .h_full()
            .when(vm.installed(), |el| {
                el.cursor(CursorStyle::PointingHand)
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
            })
            .child(vendor_icon)
            .child(name_label)
            .into_any_element();

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
            let cancel_id = id.clone();
            div()
                .flex()
                .items_center()
                .gap_1p5()
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.text_muted)
                        .child(format!("{:.0}%", vm.progress * 100.0)),
                )
                .child(
                    IconButton::new(
                        SharedString::from(format!("tog-{}", vm.id)),
                        if vm.paused() { Icon::Download } else { Icon::Pause },
                    )
                    .icon_size(14.)
                    .hit_size(26.)
                    .color(theme.text)
                    .background(theme.bg_hover)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        let id = toggle_id.clone();
                        this.store.update(cx, |s, cx| s.toggle_download(id, cx));
                    })),
                )
                .child(
                    IconButton::new(
                        SharedString::from(format!("cancel-{}", vm.id)),
                        Icon::Close,
                    )
                    .icon_size(14.)
                    .hit_size(26.)
                    .color(theme.text)
                    .background(theme.bg_hover)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        let id = cancel_id.clone();
                        this.store.update(cx, |s, cx| s.delete(id, cx));
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

        let active_dl = vm.downloading() || vm.paused();
        let track = theme.bg_hover;
        let fill = theme.text;
        // Horizontal content (icon/name, size + quantization columns, controls).
        let content = div()
            .flex()
            .items_center()
            .gap_4()
            .h(px(52.))
            .w_full()
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
            .child(action);

        div()
            // Stable id so hover state persists and updates on mouse-move
            // (not only on scroll-triggered repaints).
            .id(SharedString::from(format!("model-{}", vm.id)))
            .flex()
            .flex_col()
            .px_4()
            .rounded_lg()
            .cursor(CursorStyle::PointingHand)
            // Mirai-quantized rows get a full green tint (mirai-chat parity),
            // not just a left edge.
            .when(vm.is_mirai, |el| el.bg(theme.success.opacity(0.12)))
            .hover(move |s| s.bg(hover))
            .child(content)
            // Active downloads grow a thin progress bar along the bottom edge.
            .when(active_dl, |el| {
                el.child(
                    div().w_full().pb(px(6.)).child(
                        div()
                            .w_full()
                            .h(px(4.))
                            .rounded_full()
                            .bg(track)
                            .child(
                                div()
                                    .h_full()
                                    .w(relative(vm.progress.clamp(0., 1.)))
                                    .rounded_full()
                                    .bg(fill),
                            ),
                    ),
                )
            })
    }

    fn recommended_row(&self, cx: &mut Context<Self>, vm: &ModelVm, family_key: &str) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let key = family_key.to_string();
        let name = vm.name.clone();
        div()
            .id("recommended-model")
            .flex()
            .items_center()
            .gap_3()
            .h(px(56.))
            .px_4()
            .mb_2()
            .rounded_lg()
            .border_1()
            .border_color(theme.success.opacity(0.45))
            .bg(theme.success.opacity(0.08))
            .cursor(CursorStyle::PointingHand)
            .hover(move |s| s.bg(hover))
            .on_click(cx.listener(move |this, _, _, cx| this.open_family(key.clone(), cx)))
            .child(
                div()
                    .flex_1()
                    .flex()
                    .flex_col()
                    .gap_0p5()
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme.success)
                            .child("Recommended for your device"),
                    )
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(theme.text)
                            .child(name),
                    ),
            )
            .child(IconEl::new(Icon::ChevronRight, theme.text_muted).size(crate::tokens::icon::MD))
    }

    fn sort_control(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let label = self.sort.label();
        let mut menu = div().flex().flex_col().gap_0p5();
        for sort in [ModelSort::Size, ModelSort::Name, ModelSort::Newest] {
            let active = self.sort == sort;
            menu = menu.child(
                div()
                    .id(SharedString::from(format!("sort-{label}", label = sort.label())))
                    .px_3()
                    .py_1p5()
                    .rounded_md()
                    .text_sm()
                    .text_color(if active { theme.text } else { theme.text_muted })
                    .cursor(CursorStyle::PointingHand)
                    .hover(move |s| s.bg(hover))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.sort = sort;
                        this.sort_open = false;
                        cx.notify();
                    }))
                    .child(sort.label()),
            );
        }
        div()
            .relative()
            .child(
                div()
                    .id("sort-trigger")
                    .flex()
                    .items_center()
                    .gap_1()
                    .h(px(32.))
                    .px_2p5()
                    .rounded_lg()
                    .border_1()
                    .border_color(theme.border)
                    .bg(theme.card)
                    .cursor(CursorStyle::PointingHand)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.sort_open = !this.sort_open;
                        cx.notify();
                    }))
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.text_muted)
                            .child(format!("Sort: {label}")),
                    )
                    .child(IconEl::new(Icon::ChevronDown, theme.text_muted).size(crate::tokens::icon::SM)),
            )
            .when(self.sort_open, |el| {
                el.child(
                    div()
                        .absolute()
                        .top(px(36.))
                        .right_0()
                        .w(px(140.))
                        .p_1()
                        .rounded_lg()
                        .border_1()
                        .border_color(theme.border)
                        .bg(theme.card)
                        .shadow_md()
                        .child(menu),
                )
            })
    }
}

impl Render for LocalModelsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let query = self.search.read(cx).text().to_lowercase();
        // Take the memoized family list (rebuilt only on store/recommend/theme
        // change) and restore it before returning, so render does no per-frame
        // catalog rebuild — only the cheap query filter + sort below.
        self.ensure_families(cx);
        let cache = self.families_cache.take().expect("ensure_families populates cache");
        let families: &[FamilyVm] = &cache.1;
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
                    .map(|f| (f.name.clone(), f.vendor.clone(), f.icon_url.clone()))
                    .unwrap_or_else(|| ("Models".to_string(), String::new(), None));
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        IconButton::new("models-back", Icon::ChevronLeft)
                            .color(theme.text_muted)
                            .on_click(cx.listener(|this, _, _, cx| this.back_to_families(cx))),
                    )
                    .child(VendorIcon::new(title.1.clone()).size(crate::tokens::icon::XXL).icon_url(title.2.clone()))
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
            None => {
                let mut header_col = div()
                    .flex()
                    .flex_col()
                    .gap_0p5()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(IconEl::new(Icon::Devices, theme.text).size(crate::tokens::icon::XXL))
                            .child(
                                div()
                                    .text_xl()
                                    .font_weight(FontWeight::MEDIUM)
                                    .child("Choose local model to chat"),
                            ),
                    );
                if !self.device_label.is_empty() {
                    header_col = header_col.child(
                        div()
                            .pl(px(30.))
                            .text_sm()
                            .text_color(theme.text_muted)
                            .child(self.device_label.clone()),
                    );
                }
                header_col.into_any_element()
            }
        };

        let recommended = self.recommended_id.as_ref().and_then(|id| {
            families.iter().find_map(|f| {
                f.models
                    .iter()
                    .find(|m| &m.id == id)
                    .map(|vm| (vm.clone(), f.key.clone()))
            })
        });

        // Body list.
        let mut list = div().flex().flex_col().gap_2().pb_6();
        match &selected {
            None => {
                if let Some((vm, key)) = recommended {
                    list = list.child(self.recommended_row(cx, &vm, &key));
                }
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
                    let mut matched: Vec<ModelVm> = fam
                        .models
                        .iter()
                        .filter(|m| query.is_empty() || m.name.to_lowercase().contains(&query))
                        .cloned()
                        .collect();
                    self.sort_models(&mut matched);
                    let installed: Vec<&ModelVm> =
                        matched.iter().filter(|m| m.installed()).collect();
                    let available: Vec<&ModelVm> =
                        matched.iter().filter(|m| !m.installed()).collect();

                    let vendor = fam.vendor.clone();
                    let icon_url = fam.icon_url.clone();
                    if !installed.is_empty() {
                        list = list.child(section_header("Installed models", &theme));
                        for vm in installed {
                            list =
                                list.child(self.model_row(cx, vm, &vendor, icon_url.as_deref()));
                        }
                    }
                    if !available.is_empty() {
                        list = list.child(section_header("Available models", &theme));
                        for vm in available {
                            list =
                                list.child(self.model_row(cx, vm, &vendor, icon_url.as_deref()));
                        }
                    }
                }
            }
        }

        // `families` is fully consumed above; restore the cache for next frame.
        self.families_cache = Some(cache);

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
                                    .when(selected.is_some(), |el| {
                                        el.child(self.sort_control(cx))
                                    })
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

pub(crate) fn format_size(bytes: i64) -> String {
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
