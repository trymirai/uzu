use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, RwLock},
};

use crate::{
    TransformError,
    regex::{RegexCaptures, RegexEngine},
};

// Patterns come from schemas, but nothing forces that, so the cache is bounded.
const CACHE_LIMIT: usize = 256;

static STANDARD: LazyLock<RwLock<HashMap<String, Arc<Regex>>>> = LazyLock::new(Default::default);
static EXTENDED: LazyLock<RwLock<HashMap<String, Arc<Regex>>>> = LazyLock::new(Default::default);

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

    /// Compiles a pattern once per process instead of once per call.
    pub fn cached(
        pattern: &str,
        engine: &RegexEngine,
    ) -> Result<Arc<Self>, TransformError> {
        let cache = match engine {
            RegexEngine::Standard => &STANDARD,
            RegexEngine::Extended => &EXTENDED,
        };

        if let Some(regex) = cache.read().unwrap().get(pattern) {
            return Ok(regex.clone());
        }

        let regex = Arc::new(Self::new(pattern, engine)?);
        let mut cache = cache.write().unwrap();
        if cache.len() < CACHE_LIMIT {
            cache.insert(pattern.to_string(), regex.clone());
        }
        Ok(regex)
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
