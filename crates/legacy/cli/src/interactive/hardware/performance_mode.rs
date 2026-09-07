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

    pub fn cycled(
        self,
        delta: i64,
    ) -> Self {
        match (self as i64 + delta.rem_euclid(2)) % 2 {
            0 => Self::Auto,
            _ => Self::Fast,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PerformanceMode;

    #[test]
    fn modes_cycle_in_both_directions_without_overflow() {
        let mut mode = PerformanceMode::default();
        for expected in [PerformanceMode::Fast, PerformanceMode::Auto] {
            mode = mode.cycled(1);
            assert_eq!(mode, expected);
        }
        for expected in [PerformanceMode::Fast, PerformanceMode::Auto] {
            mode = mode.cycled(-1);
            assert_eq!(mode, expected);
        }
        assert_eq!(PerformanceMode::Auto.cycled(i64::MIN), PerformanceMode::Auto);
        assert_eq!(PerformanceMode::Fast.cycled(i64::MAX), PerformanceMode::Auto);
    }
}
