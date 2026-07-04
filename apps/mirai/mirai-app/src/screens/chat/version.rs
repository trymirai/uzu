use std::cell::RefCell;

use crate::components::markdown::ParsedMarkdown;

#[derive(Clone, Default)]
pub(super) struct Version {
    pub text: String,
    pub reasoning: Option<String>,
    pub tps: Option<f32>,
    pub tokens: Option<u32>,
    pub ttft: Option<f32>,
    pub total_time: Option<f32>,
    pub error: bool,
    pub model_name: Option<String>,
    pub parsed: RefCell<Option<(String, ParsedMarkdown)>>,
}

impl Version {
    pub(super) fn parsed_markdown(&self) -> std::cell::Ref<'_, ParsedMarkdown> {
        {
            let mut cache = self.parsed.borrow_mut();
            let stale = cache.as_ref().map(|(t, _)| t != &self.text).unwrap_or(true);
            if stale {
                *cache = Some((self.text.clone(), crate::components::markdown::parse(&self.text)));
            }
        }
        std::cell::Ref::map(self.parsed.borrow(), |c| &c.as_ref().unwrap().1)
    }
}
