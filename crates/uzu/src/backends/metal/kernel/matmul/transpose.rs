#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum TransposeConfiguration {
    NN,
    NT,
    TN,
    TT,
}

impl TransposeConfiguration {
    pub fn as_str(&self) -> &'static str {
        match self {
            TransposeConfiguration::NN => "nn",
            TransposeConfiguration::NT => "nt",
            TransposeConfiguration::TN => "tn",
            TransposeConfiguration::TT => "tt",
        }
    }
}

pub(super) fn transpose_configuration(
    a_t: bool,
    b_t: bool,
) -> TransposeConfiguration {
    match (a_t, b_t) {
        (false, false) => TransposeConfiguration::NN,
        (false, true) => TransposeConfiguration::NT,
        (true, false) => TransposeConfiguration::TN,
        (true, true) => TransposeConfiguration::TT,
    }
}
