use std::collections::HashMap;

use iocraft::prelude::*;
use shoji::types::model::Model;

use crate::{
    cli::{
        components::{ApplicationState, Loading, Selector, SelectorItem, SelectorStyle},
        flows::{Flow, FlowEvent},
    },
    storage::types::DownloadState,
};

pub struct ModelsFlow {
    pub registry_id: Option<String>,
    pub family_id: Option<String>,
}

impl Flow for ModelsFlow {
    fn render(
        &self,
        on_event: Handler<FlowEvent>,
    ) -> AnyElement<'static> {
        element! {
            Models(
                registry_id: self.registry_id.clone(),
                family_id: self.family_id.clone(),
                on_event: on_event,
            )
        }
        .into()
    }
}

#[derive(Default, Props)]
pub struct ModelsProps {
    pub registry_id: Option<String>,
    pub family_id: Option<String>,
    pub on_event: Handler<FlowEvent>,
}

#[component]
fn Models(
    props: &mut ModelsProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let on_event = std::mem::take(&mut props.on_event);
    let registry_id = props.registry_id.clone();
    let family_id = props.family_id.clone();
    let state = *hooks.use_context::<State<ApplicationState>>();
    let mut models_state = hooks.use_state(|| None::<Vec<Model>>);
    let mut model_download_statuses_state = hooks.use_state(|| None::<HashMap<String, DownloadState>>);

    hooks.use_future({
        let engine = state.read().engine.clone();
        let registry_id = registry_id.clone();
        let family_id = family_id.clone();
        async move {
            let models: Vec<Model> = engine
                .models()
                .await
                .unwrap_or_default()
                .into_iter()
                .filter(|model| registry_id.as_ref().map_or(true, |id| &model.registry.identifier == id))
                .filter(|model| {
                    family_id
                        .as_ref()
                        .map_or(true, |id| model.family.as_ref().map_or(false, |family| &family.identifier == id))
                })
                .collect();
            let download_statuses = engine.download_states().await;

            models_state.set(Some(models));
            model_download_statuses_state.set(Some(download_statuses));
        }
    });

    let accent_color = state.read().theme.accent_color;
    let subtitle_color = state.read().theme.subtitle_color;
    let columns_padding = state.read().theme.padding_wide();

    let list = models_state.read().clone().unwrap_or_default();
    let loaded = models_state.read().is_some();
    let items: Vec<SelectorItem> = list
        .iter()
        .map(|model| {
            let download_status = model_download_statuses_state
                .read()
                .as_ref()
                .and_then(|statuses| statuses.get(&model.identifier))
                .map(|status| status.name());
            SelectorItem {
                title: model.name(),
                description: download_status,
                color: None,
            }
        })
        .collect();
    let height = (items.len() as u16).min(5).max(1);

    element! {
        Loading(loaded: loaded) {
            Selector(
                items: items,
                style: SelectorStyle::WithIcon,
                maximal_height: height,
                accent_color: accent_color,
                subtitle_color: subtitle_color,
                columns_padding: columns_padding,
                on_submit: move |index: usize| {
                    let mut state = state;
                    if let Some(model) = list.get(index) {
                        let summary = format!("Model: {}", model.name());
                        state.write().model = Some(model.clone());
                        on_event(FlowEvent::finish(summary));
                    }
                },
            )
        }
    }
}
