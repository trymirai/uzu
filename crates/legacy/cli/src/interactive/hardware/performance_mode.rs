#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PerformanceMode {
    #[default]
    Auto,
    Fast,
}

impl PerformanceMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Fast => "Fast",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Auto => "Restore the previous system settings",
            Self::Fast => "Maximum fans and macOS High Power Mode",
        }
    }

    pub fn toggled(self) -> Self {
        match self {
            Self::Auto => Self::Fast,
            Self::Fast => Self::Auto,
        }
    }
}
