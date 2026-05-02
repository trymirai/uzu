use std::sync::Arc;

use iocraft::prelude::*;
use shoji::types::model::{Model, ModelRegistry};

use super::{Flow, FlowEvent, ModelFamiliesFlow, ModelsFlow};
use crate::cli::components::{ApplicationState, Loading, Selector, SelectorItem, SelectorStyle};

pub struct ModelRegistriesFlow;

impl Flow for ModelRegistriesFlow {
    fn render(
        &self,
        on_event: Handler<FlowEvent>,
    ) -> AnyElement<'static> {
        element! { ModelRegistries(on_event: on_event) }.into()
    }
}

#[derive(Default, Props)]
pub struct ModelRegistriesProps {
    pub on_event: Handler<FlowEvent>,
}

#[component]
fn ModelRegistries(
    props: &mut ModelRegistriesProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let on_event = Arc::new(std::mem::take(&mut props.on_event));
    let state = *hooks.use_context::<State<ApplicationState>>();
    let mut registries_state = hooks.use_state(|| None::<Vec<ModelRegistry>>);

    let on_pick = hooks.use_async_handler({
        let on_event = on_event.clone();
        let engine = state.read().engine.clone();
        move |registry: ModelRegistry| {
            let on_event = on_event.clone();
            let engine = engine.clone();
            async move {
                let registry_models: Vec<Model> = engine
                    .models()
                    .await
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|model| model.registry.identifier == registry.identifier)
                    .collect();
                let next: Box<dyn Flow> = if registry_models.iter().any(|model| model.family.is_none()) {
                    Box::new(ModelsFlow {
                        registry_id: Some(registry.identifier.clone()),
                        family_id: None,
                    })
                } else {
                    Box::new(ModelFamiliesFlow {
                        registry_id: Some(registry.identifier.clone()),
                    })
                };
                (*on_event)(FlowEvent::transition(format!("Registry: {}", registry.name()), next));
            }
        }
    });

    hooks.use_future({
        let on_pick = on_pick.clone();
        let engine = state.read().engine.clone();
        async move {
            let registries = engine.model_registries().await.unwrap_or_default();
            if registries.len() == 1 {
                on_pick(registries[0].clone());
            } else {
                registries_state.set(Some(registries));
            }
        }
    });

    let accent_color = state.read().theme.accent_color;
    let subtitle_color = state.read().theme.subtitle_color;
    let columns_padding = state.read().theme.padding_wide();

    let list = registries_state.read().clone().unwrap_or_default();
    let loaded = registries_state.read().is_some();
    let items: Vec<SelectorItem> = list
        .iter()
        .map(|registry| SelectorItem {
            title: registry.name(),
            description: Some(registry.identifier.clone()),
            color: None,
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
                    if let Some(registry) = list.get(index) {
                        on_pick(registry.clone());
                    }
                },
            )
        }
    }
}
