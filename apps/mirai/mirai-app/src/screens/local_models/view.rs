use gpui::{Context, Entity, EventEmitter, FontWeight, IntoElement, Render, Window, div, prelude::*, px};

use super::{
    event::LocalModelsEvent, family_view_model::FamilyViewModel, format::section_header,
    model_view_model::ModelViewModel,
};
use crate::{
    components::{ConfirmModal, Icon, IconButton, IconEl, TextInput, VendorIcon},
    device_info,
    model_sort::ModelSort,
    models_store::ModelsStore,
    theme::{ActiveTheme, layout::CONTENT_MAX_WIDTH},
    tokens,
};

pub struct LocalModelsView {
    pub(super) store: Entity<ModelsStore>,
    pub(super) search: Entity<TextInput>,
    pub(super) selected_family: Option<String>,
    pub(super) confirm_delete: Option<(String, String)>,
    pub(super) device_label: String,
    pub(super) recommended_id: Option<String>,
    pub(super) sort: ModelSort,
    pub(super) sort_open: bool,

    pub(super) families_cache: Option<(bool, Vec<FamilyViewModel>)>,
}

impl EventEmitter<LocalModelsEvent> for LocalModelsView {}

impl LocalModelsView {
    pub fn new(
        store: Entity<ModelsStore>,
        cx: &mut Context<Self>,
    ) -> Self {
        let search = cx.new(|cx| TextInput::new(cx, "Search families"));

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
}

impl Render for LocalModelsView {
    fn render(
        &mut self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let query = self.search.read(cx).text().to_lowercase();

        self.ensure_families(cx);
        let cache = self.families_cache.take().expect("ensure_families populates cache");
        let families: &[FamilyViewModel] = &cache.1;
        let selected = self.selected_family.clone();

        let modal = self.confirm_delete.clone().map(|(id, name)| {
            ConfirmModal::new("Delete model", format!("Delete \"{name}\"? You can download it again later."))
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
                    .child(VendorIcon::new(title.1.clone()).size(tokens::icon::XXL).icon_url(title.2.clone()))
                    .child(div().text_xl().font_weight(FontWeight::MEDIUM).child(title.0))
                    .child(div().text_sm().text_color(theme.text_muted).child(format!("from {}", title.1)))
                    .into_any_element()
            },
            None => {
                let mut header_col = div().flex().flex_col().gap_0p5().child(
                    div()
                        .flex()
                        .items_center()
                        .gap_2()
                        .child(IconEl::new(Icon::Devices, theme.text).size(tokens::icon::XXL))
                        .child(div().text_xl().font_weight(FontWeight::MEDIUM).child("Choose local model to chat")),
                );
                if !self.device_label.is_empty() {
                    header_col = header_col.child(
                        div().pl(px(30.)).text_sm().text_color(theme.text_muted).child(self.device_label.clone()),
                    );
                }
                header_col.into_any_element()
            },
        };

        let recommended =
            families.iter().find_map(|f| f.models.iter().find(|m| m.recommended).map(|vm| (vm.clone(), f.key.clone())));

        let mut list = div().flex().flex_col().gap_2().pb_6();
        match &selected {
            None => {
                if let Some((vm, key)) = recommended {
                    list = list.child(self.recommended_row(cx, &vm, &key));
                }
                let visible: Vec<&FamilyViewModel> = families
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
            },
            Some(key) => {
                if let Some(fam) = families.iter().find(|f| &f.key == key) {
                    let mut matched: Vec<ModelViewModel> = fam
                        .models
                        .iter()
                        .filter(|m| query.is_empty() || m.name.to_lowercase().contains(&query))
                        .cloned()
                        .collect();
                    self.sort_models(&mut matched);
                    let installed: Vec<&ModelViewModel> = matched.iter().filter(|m| m.installed()).collect();
                    let available: Vec<&ModelViewModel> = matched.iter().filter(|m| !m.installed()).collect();

                    let vendor = fam.vendor.clone();
                    let icon_url = fam.icon_url.clone();
                    if !installed.is_empty() {
                        list = list.child(section_header("Installed models", &theme));
                        for vm in installed {
                            list = list.child(self.model_row(cx, vm, &vendor, icon_url.as_deref()));
                        }
                    }
                    if !available.is_empty() {
                        list = list.child(section_header("Available models", &theme));
                        for vm in available {
                            list = list.child(self.model_row(cx, vm, &vendor, icon_url.as_deref()));
                        }
                    }
                }
            },
        }

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
                        div().pt_10().pb_3().flex().items_center().justify_between().gap_4().child(header).child(
                            div()
                                .flex()
                                .items_center()
                                .gap_2()
                                .when(selected.is_some(), |el| el.child(self.sort_control(cx)))
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
                    .child(div().id("models-list").flex_1().min_h_0().overflow_y_scroll().child(list)),
            )
            .children(modal)
    }
}
