use crate::{
    TransformError,
    regex::{RegexCaptures, RegexEngine},
};

pub enum Regex {
    Standard(regex::Regex),
    Extended(fancy_regex::Regex),
}

impl Regex {
    pub fn new(
        pattern: &str,
        engine: &RegexEngine,
    ) -> Result<Self, TransformError> {
        match engine {
            RegexEngine::Standard => {
                regex::Regex::new(pattern).map(Regex::Standard).map_err(|_| TransformError::InvalidRegex {
                    pattern: pattern.to_string(),
                })
            },
            RegexEngine::Extended => {
                fancy_regex::Regex::new(pattern).map(Regex::Extended).map_err(|_| TransformError::InvalidRegex {
                    pattern: pattern.to_string(),
                })
            },
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Regex::Standard(regex) => regex.as_str(),
            Regex::Extended(regex) => regex.as_str(),
        }
    }

    pub fn captures(
        &self,
        text: &str,
    ) -> Option<RegexCaptures> {
        match self {
            Regex::Standard(regex) => regex.captures(text).map(|captures| RegexCaptures::from_standard(&captures)),
            Regex::Extended(regex) => {
                regex.captures(text).ok().flatten().map(|captures| RegexCaptures::from_extended(&captures))
            },
        }
    }

    pub fn captures_iter(
        &self,
        text: &str,
    ) -> Vec<RegexCaptures> {
        match self {
            Regex::Standard(regex) => {
                regex.captures_iter(text).map(|captures| RegexCaptures::from_standard(&captures)).collect()
            },
            Regex::Extended(regex) => regex
                .captures_iter(text)
                .filter_map(|result| result.ok())
                .map(|captures| RegexCaptures::from_extended(&captures))
                .collect(),
        }
    }

    pub fn replace_all(
        &self,
        text: &str,
        replacement: &str,
    ) -> String {
        match self {
            Regex::Standard(regex) => regex.replace_all(text, replacement).to_string(),
            Regex::Extended(regex) => regex.replace_all(text, replacement).to_string(),
        }
    }
}
