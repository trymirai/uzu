use std::cmp::Ordering;

use futures::{StreamExt, channel::mpsc};
use gpui::Context;
use gpui_tokio::Tokio;
use uzu::types::{model::Model, session::classification::ClassificationMessage};

use super::{classification_outcome::ClassificationOutcome, view::RoutersView};
use crate::engine;

impl RoutersView {
    pub(super) fn clear(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        self.input.update(cx, |input, cx| input.set_text("", cx));

        self.classify_gen = self.classify_gen.wrapping_add(1);
        self.classifying = false;
        self.result = None;
        self.error = None;
        cx.notify();
    }

    pub(super) fn resolved_router(
        &self,
        cx: &Context<Self>,
    ) -> Option<Model> {
        self.store.read(cx).resolve_installed(self.selected.as_ref())
    }

    fn classify(
        &mut self,
        text: String,
        cx: &mut Context<Self>,
    ) {
        if self.classifying {
            return;
        }
        let text = text.trim().to_string();
        if text.is_empty() {
            return;
        }
        let Some(model) = self.resolved_router(cx) else {
            self.error = Some("Download and select a router first.".to_string());
            cx.notify();
            return;
        };
        self.selected = Some(model.clone());
        self.classifying = true;
        self.error = None;
        self.result = None;
        self.classify_gen = self.classify_gen.wrapping_add(1);
        let gen_id = self.classify_gen;
        cx.notify();

        let Some(engine) = engine::try_engine(cx) else {
            self.classifying = false;
            self.error = Some("engine unavailable".to_string());
            cx.notify();
            return;
        };

        let (tx, mut rx) = mpsc::unbounded::<ClassificationOutcome>();
        Tokio::spawn(cx, async move {
            match engine.classification(model).await {
                Ok(session) => match session.classify(vec![ClassificationMessage::user(text)]).await {
                    Ok(output) => {
                        let mut values: Vec<(String, f64)> = output.probabilities.values.into_iter().collect();
                        values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                        let _ = tx.unbounded_send(ClassificationOutcome::Ok(values));
                    },
                    Err(err) => {
                        let _ = tx.unbounded_send(ClassificationOutcome::Err(format!("{err:?}")));
                    },
                },
                Err(err) => {
                    let _ = tx.unbounded_send(ClassificationOutcome::Err(err.to_string()));
                },
            }
        })
        .detach();

        cx.spawn(async move |this, cx| {
            while let Some(msg) = rx.next().await {
                let keep = this.update(cx, |view, cx| {
                    if view.classify_gen != gen_id {
                        return false;
                    }
                    match msg {
                        ClassificationOutcome::Ok(values) => view.result = Some(values),
                        ClassificationOutcome::Err(err) => view.error = Some(err),
                    }
                    view.classifying = false;
                    cx.notify();
                    true
                });
                if !matches!(keep, Ok(true)) {
                    break;
                }
            }
        })
        .detach();
    }

    pub(super) fn classify_from_button(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        let text = self.input.read(cx).text();
        self.classify(text, cx);
    }
}
