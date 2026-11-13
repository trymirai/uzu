use std::sync::Arc;

use iocraft::prelude::*;

use crate::interactive::{
    components::{ApplicationState, Selector, SelectorItem, SelectorStyle},
    flows::{DeleteModelFlowViewProps, FlowEvent},
};

#[component]
pub fn DeleteModelFlowView(
    props: &mut DeleteModelFlowViewProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let on_event = Arc::new(std::mem::take(&mut props.on_event));
    let state = *hooks.use_context::<State<ApplicationState>>();
    let model = state.read().model_state.as_ref().map(|model_state| model_state.model.clone());
    let name = model.as_ref().map(|model| model.name()).unwrap_or_default();

    let on_choice = hooks.use_async_handler({
        let engine = state.read().engine.clone();
        move |delete: bool| {
            let engine = engine.clone();
            let model = model.clone();
            let on_event = on_event.clone();
            async move {
                let Some(model) = model else {
                    return;
                };
                let result = if !delete {
                    format!("Kept {}", model.name())
                } else {
                    match engine.downloader(&model).delete().await {
                        Ok(()) => format!("Deleted {}", model.name()),
                        Err(error) => format!("Unable to delete {}: {error}", model.name()),
                    }
                };
                on_event(FlowEvent::finish(result));
            }
        }
    });

    let theme = state.read().theme().clone();
    let items = ["Keep it", "Delete it"]
        .map(|title| SelectorItem {
            title: title.to_string(),
            description: None,
            color: None,
        })
        .to_vec();

    element! {
        View(flex_direction: FlexDirection::Column) {
            Text(content: format!("Delete {name} from this device?"), weight: Weight::Bold)
            Selector(
                items: items,
                style: SelectorStyle::Plain,
                maximal_height: 2u16,
                accent_color: theme.accent_color,
                subtitle_color: theme.subtitle_color,
                columns_padding: theme.padding_wide(),
                on_submit: move |index: usize| on_choice(index == 1),
            )
        }
    }
}
