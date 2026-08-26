use std::{collections::HashMap, sync::Arc};

use iocraft::prelude::*;
use shoji::types::{basic::SamplingParameters, model::Model};
use uzu::storage::types::DownloadState;

use crate::{
    common::thinking::ThinkingSupport,
    interactive::{
        components::{ApplicationState, Loading, ModelState, Selector, SelectorItem, SelectorStyle},
        flows::{Flow, FlowEvent},
    },
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
    let on_event = Arc::new(std::mem::take(&mut props.on_event));
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
                .filter(|model| registry_id.as_ref().is_none_or(|id| &model.registry.identifier == id))
                .filter(|model| {
                    family_id
                        .as_ref()
                        .is_none_or(|id| model.family.as_ref().is_some_and(|family| &family.identifier == id))
                })
                .collect();
            let download_statuses = engine.download_states().await;

            models_state.set(Some(models));
            model_download_statuses_state.set(Some(download_statuses));
        }
    });

    let accent_color = state.read().theme().accent_color;
    let subtitle_color = state.read().theme().subtitle_color;
    let columns_padding = state.read().theme().padding_wide();

    let list = models_state.read().clone().unwrap_or_default();
    let download_statuses = model_download_statuses_state.read().clone().unwrap_or_default();
    let loaded = models_state.read().is_some();
    let items: Vec<SelectorItem> = list
        .iter()
        .map(|model| {
            let download_status = download_statuses.get(&model.identifier).map(|status| status.name());
            SelectorItem {
                title: model.name(),
                description: download_status,
                color: None,
            }
        })
        .collect();
    let height = (items.len() as u16).clamp(1, 5);

    let on_pick = hooks.use_async_handler({
        let engine = state.read().engine.clone();
        move |model: Model| {
            let engine = engine.clone();
            let on_event = on_event.clone();
            let mut state = state;
            async move {
                let download_state =
                    engine.download_state(&model).await.unwrap_or_else(|| DownloadState::not_downloaded(0));
                let preferences_result = {
                    let mut app_state = state.write();
                    app_state.model_state = Some(ModelState {
                        model: model.clone(),
                        download_state,
                        session_state: None,
                        thinking: ThinkingSupport::default(),
                        sampling_defaults: SamplingParameters::default(),
                        thinking_locked: false,
                    });
                    let mut preferences = app_state.preferences().clone();
                    preferences.selected_model_id = Some(model.identifier.clone());
                    app_state.set_preferences(&preferences)
                };
                let result = match preferences_result {
                    Ok(()) => format!("Model: {}", model.name()),
                    Err(error) => format!("Model: {}, unable to save preference: {}", model.name(), error),
                };
                (*on_event)(FlowEvent::finish(result));
            }
        }
    });

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
                    if let Some(model) = list.get(index) {
                        on_pick(model.clone());
                    }
                },
            )
        }
    }
}
