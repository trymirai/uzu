use indexmap::IndexSet;
use iocraft::prelude::*;
use shoji::types::model::ModelFamily;

use crate::cli::{
    components::{ApplicationState, Loading, Selector, SelectorItem, SelectorStyle},
    flows::{Flow, FlowEvent, ModelsFlow},
};

pub struct ModelFamiliesFlow {
    pub registry_id: Option<String>,
}

impl Flow for ModelFamiliesFlow {
    fn render(
        &self,
        on_event: Handler<FlowEvent>,
    ) -> AnyElement<'static> {
        element! {
            ModelFamilies(
                registry_id: self.registry_id.clone(),
                on_event: on_event,
            )
        }
        .into()
    }
}

#[derive(Default, Props)]
pub struct ModelFamiliesProps {
    pub registry_id: Option<String>,
    pub on_event: Handler<FlowEvent>,
}

#[component]
fn ModelFamilies(
    props: &mut ModelFamiliesProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let on_event = std::mem::take(&mut props.on_event);
    let registry_id = props.registry_id.clone();
    let state = *hooks.use_context::<State<ApplicationState>>();
    let mut families = hooks.use_state(|| None::<Vec<ModelFamily>>);

    hooks.use_future({
        let engine = state.read().engine.clone();
        let registry_id = registry_id.clone();
        async move {
            let loaded = match registry_id {
                Some(identifier) => {
                    let mut seen: IndexSet<String> = IndexSet::new();
                    engine
                        .models()
                        .await
                        .unwrap_or_default()
                        .into_iter()
                        .filter(|model| model.registry.identifier == identifier)
                        .filter_map(|model| model.family.clone())
                        .filter(|family| seen.insert(family.identifier.clone()))
                        .collect()
                },
                None => engine.model_families().await.unwrap_or_default(),
            };
            families.set(Some(loaded));
        }
    });

    let accent_color = state.read().theme.accent_color;
    let subtitle_color = state.read().theme.subtitle_color;
    let columns_padding = state.read().theme.padding_wide();

    let list = families.read().clone().unwrap_or_default();
    let loaded = families.read().is_some();
    let items: Vec<SelectorItem> = list
        .iter()
        .map(|family| SelectorItem {
            title: family.name(),
            description: Some(family.vendor.metadata.name.clone()),
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
                    if let Some(family) = list.get(index) {
                        let name = format!("Family: {} ({})", family.name(), family.vendor.name());
                        let next: Box<dyn Flow> = Box::new(ModelsFlow {
                            registry_id: registry_id.clone(),
                            family_id: Some(family.identifier.clone()),
                        });
                        on_event(FlowEvent::transition(name, next));
                    }
                },
            )
        }
    }
}
