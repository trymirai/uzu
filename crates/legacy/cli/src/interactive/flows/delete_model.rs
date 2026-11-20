use iocraft::prelude::*;
use shoji::types::model::Model;

use crate::interactive::{
    components::{ApplicationState, HistoryCellType, Selector, SelectorItem, SelectorStyle},
    flows::{Flow, FlowEvent},
};

pub struct DeleteModelFlow {
    pub model: Model,
}

impl Flow for DeleteModelFlow {
    fn render(
        &self,
        on_event: Handler<FlowEvent>,
    ) -> AnyElement<'static> {
        element! { DeleteModel(model: self.model.clone(), on_event: on_event) }.into()
    }
}

#[derive(Default, Props)]
pub struct DeleteModelProps {
    pub model: Option<Model>,
    pub on_event: Handler<FlowEvent>,
}

#[component]
fn DeleteModel(
    props: &mut DeleteModelProps,
    hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let on_event = std::mem::take(&mut props.on_event);
    let model = props.model.clone().expect("the delete flow passes its model");
    let name = model.name();
    let state = *hooks.use_context::<State<ApplicationState>>();
    let choices = [("Keep it", false), ("Delete it", true)];

    let on_choice = {
        let name = name.clone();
        move |delete: bool| {
            if !delete {
                on_event(FlowEvent::finish(format!("Kept {name}")));
                return;
            }
            on_event(FlowEvent::finish(format!("Deleting {name}")));
            let engine = state.read().engine.clone();
            let model = model.clone();
            let name = name.clone();
            // The delete outlives the prompt, which closes at once, so it reports its own result.
            tokio::spawn(async move {
                let result = match engine.downloader(&model).delete().await {
                    Ok(()) => format!("Deleted {name}"),
                    Err(error) => format!("Unable to delete {name}: {error}"),
                };
                let mut state = state;
                if let Some(mut state) = state.try_write() {
                    state.history.push(HistoryCellType::CommandResult {
                        result,
                    });
                }
            });
        }
    };

    let theme = state.read().theme().clone();
    let items = choices
        .map(|(title, _)| SelectorItem {
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
                maximal_height: choices.len() as u16,
                accent_color: theme.accent_color,
                subtitle_color: theme.subtitle_color,
                columns_padding: theme.padding_wide(),
                on_submit: move |index: usize| on_choice(choices[index].1),
            )
        }
    }
}
