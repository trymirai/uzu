pub const HINT_COMMANDS: &str = "/ for commands";
pub const HINT_SEND: &str = "enter to send";
// An example input, line by line, with what it is and a note on how it's written.
pub struct HintExample {
    pub label: &'static str,
    pub lines: &'static [&'static str],
    pub note: &'static str,
}

pub const HINT_DECISION_STATE_TITLE: &str = "Write the text you want to ask questions about, then press enter";
pub const HINT_DECISION_STATE_EXAMPLES: &[HintExample] = &[HintExample {
    label: "for example",
    lines: &["This LLM runs right on my laptop and answers in under a second, love it!"],
    note: "a message, a document or JSON",
}];
pub const HINT_DECISION_STATE_NOTES: &[&str] = &["ctrl+j, or \\ then enter, starts a new line"];
pub const HINT_DECISION_QUESTION_TITLE: &str =
    "Ask a question about the text, then press enter. Each question is sent on its own";
pub const HINT_DECISION_QUESTION_EXAMPLES: &[HintExample] = &[
    HintExample {
        label: "yes or no",
        lines: &["Does the user like the model?"],
        note: "just the question",
    },
    HintExample {
        label: "choice",
        lines: &["What does the user praise?", "speed: how fast it answers", "accuracy", "price"],
        note: "one option per line, a description after : is optional",
    },
    HintExample {
        label: "score",
        lines: &["How enthusiastic is the user?", "Low < Medium < High"],
        note: "levels on one line, from lowest to highest",
    },
];
pub const HINT_DECISION_QUESTION_NOTES: &[&str] =
    &["esc to change the text", "ctrl+j, or \\ then enter, starts a new line"];
pub const HINT_STORAGE_PAUSE_RESUME: &str = "ctrl+p to pause/resume download";
pub const HINT_STORAGE_DELETE: &str = "ctrl+d to delete model";
pub const HINT_SESSION_INTERRUPT: &str = "esc to interrupt";
