#[derive(Debug, PartialEq, Eq)]
pub(super) enum ToolChoiceError {
    Invalid(String),
    Unsupported(String),
    UnknownFunction(String),
}

impl ToolChoiceError {
    pub(super) fn message(&self) -> String {
        match self {
            ToolChoiceError::Invalid(detail) => format!("tool_choice is not recognized: {detail}"),
            ToolChoiceError::Unsupported(choice) => {
                format!("tool_choice {choice:?} is not supported by this server")
            },
            ToolChoiceError::UnknownFunction(name) => {
                format!("tool_choice refers to function {name:?}, which is not present in tools")
            },
        }
    }

    pub(super) fn code(&self) -> &'static str {
        match self {
            ToolChoiceError::Invalid(_) | ToolChoiceError::UnknownFunction(_) => "invalid_tool_choice",
            ToolChoiceError::Unsupported(_) => "unsupported_tool_choice",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum ToolDefinitionError {
    ToolsUnsupported,
    StrictUnsupported {
        index: usize,
    },
    UnsupportedKind {
        index: usize,
        kind: String,
    },
}

impl ToolDefinitionError {
    pub(super) fn message(&self) -> String {
        match self {
            ToolDefinitionError::ToolsUnsupported => "tool calls are not supported by the loaded model".to_string(),
            ToolDefinitionError::StrictUnsupported {
                ..
            } => "strict function tools are not supported by this server".to_string(),
            ToolDefinitionError::UnsupportedKind {
                kind,
                ..
            } => format!("tool type {kind:?} is not supported by this server"),
        }
    }

    pub(super) fn code(&self) -> &'static str {
        match self {
            ToolDefinitionError::ToolsUnsupported => "unsupported_tools",
            ToolDefinitionError::StrictUnsupported {
                ..
            } => "unsupported_strict_tool",
            ToolDefinitionError::UnsupportedKind {
                ..
            } => "unsupported_tool_type",
        }
    }

    pub(super) fn param(&self) -> String {
        match self {
            ToolDefinitionError::ToolsUnsupported => "tools".to_string(),
            ToolDefinitionError::StrictUnsupported {
                index,
            } => format!("tools[{index}].function.strict"),
            ToolDefinitionError::UnsupportedKind {
                index,
                ..
            } => format!("tools[{index}].type"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct ParallelToolCallsError;

impl ParallelToolCallsError {
    pub(super) fn message(&self) -> String {
        "parallel_tool_calls: false is not supported by this server when tools are enabled".to_string()
    }

    pub(super) fn code(&self) -> &'static str {
        "unsupported_parallel_tool_calls"
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct ToolHistoryError {
    pub message: String,
    pub param: String,
    pub code: &'static str,
}

impl ToolHistoryError {
    pub(super) fn message(&self) -> String {
        self.message.clone()
    }

    pub(super) fn code(&self) -> &'static str {
        self.code
    }

    pub(super) fn param(&self) -> String {
        self.param.clone()
    }
}
