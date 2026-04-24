#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Language {
    Swift,
    TS,
}

impl Language {
    pub fn to_string(&self) -> String {
        match self {
            Language::Swift => "swift".to_string(),
            Language::TS => "ts".to_string(),
        }
    }
}
