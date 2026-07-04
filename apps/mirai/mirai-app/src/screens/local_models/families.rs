use std::{cmp::Ordering, collections::HashMap};

use gpui::Context;

use super::{
    family_view_model::FamilyViewModel,
    format::{format_params, format_size, quant_label},
    model_view_model::ModelViewModel,
    view::LocalModelsView,
};
use crate::{
    model_recommend,
    model_sort::{self, ModelSort},
    theme::ActiveTheme,
};

impl LocalModelsView {
    pub(super) fn spawn_recommend(cx: &mut Context<Self>) {
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

    pub fn open_family(
        &mut self,
        key: String,
        cx: &mut Context<Self>,
    ) {
        self.selected_family = Some(key);
        self.search.update(cx, |i, cx| i.clear(cx));
        cx.notify();
    }

    pub(super) fn back_to_families(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        self.selected_family = None;
        self.search.update(cx, |i, cx| i.clear(cx));
        cx.notify();
    }

    fn families(
        &self,
        cx: &Context<Self>,
    ) -> Vec<FamilyViewModel> {
        let dark = cx.theme().dark;
        let store = self.store.read(cx);
        let recommended_id = self.recommended_id.as_deref();
        let mut order: Vec<String> = Vec::new();
        let mut families: HashMap<String, FamilyViewModel> = Default::default();
        let mut recommended_family: Option<String> = None;

        for row in &store.rows {
            let (key, name, vendor) = match &row.model.family {
                Some(f) => (f.identifier.clone(), f.name(), f.vendor.name()),
                None => ("other".to_string(), "Other".to_string(), String::new()),
            };

            let is_recommended = recommended_id.is_some_and(|rid| row.model.repo_ids().iter().any(|r| r == rid));
            if is_recommended {
                recommended_family = Some(key.clone());
            }
            let quant = quant_label(&row.model);
            let is_mirai = row
                .model
                .quantization
                .as_ref()
                .map(|q| q.method.to_lowercase().contains("mirai") || q.identifier.to_lowercase().contains("mirai"))
                .unwrap_or(false);
            let bytes = row.display_size_bytes();
            let installed_at = store.installed_at(row.id());
            let vm = ModelViewModel {
                id: row.id().to_string(),
                name: row.name(),
                size: format_size(bytes),
                bytes,
                quant,
                phase: row.phase(),
                progress: row.progress(),
                is_mirai,
                recommended: is_recommended,
            };
            let entry = families.entry(key.clone()).or_insert_with(|| {
                order.push(key.clone());
                FamilyViewModel {
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
            let mut params: Vec<f64> = fam.models.iter().filter_map(|m| model_sort::parse_params(&m.name)).collect();
            params.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            if let (Some(min), Some(max)) = (params.first(), params.last()) {
                fam.range = Some(if (min - max).abs() < f64::EPSILON {
                    format_params(*max)
                } else {
                    format!("{} – {}", format_params(*min), format_params(*max))
                });
            }
        }

        let mut list: Vec<FamilyViewModel> = order.into_iter().filter_map(|k| families.remove(&k)).collect();

        let rec_key = recommended_family;
        list.sort_by(|a, b| {
            let rank = |f: &FamilyViewModel| -> u8 {
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
            rank(a).cmp(&rank(b)).then_with(|| match (rank(a), rank(b)) {
                (2, 2) => b.last_installed_at.cmp(&a.last_installed_at),
                _ => a.name.cmp(&b.name),
            })
        });
        list
    }

    pub(super) fn ensure_families(
        &mut self,
        cx: &Context<Self>,
    ) {
        let dark = cx.theme().dark;
        let valid = matches!(&self.families_cache, Some((d, _)) if *d == dark);
        if !valid {
            let built = self.families(cx);
            self.families_cache = Some((dark, built));
        }
    }

    pub(super) fn sort_models(
        &self,
        models: &mut [ModelViewModel],
    ) {
        match self.sort {
            ModelSort::Size => {
                models.sort_by(|a, b| a.bytes.cmp(&b.bytes).then_with(|| model_sort::sort_by_name(&a.name, &b.name)))
            },
            ModelSort::Name => models.sort_by(|a, b| model_sort::sort_by_name(&a.name, &b.name)),
            ModelSort::Newest => models.sort_by(|a, b| model_sort::sort_by_newest(&a.name, &b.name)),
        }
    }
}
