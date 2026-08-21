use crate::regex::RegexMatch;

#[derive(Debug, Clone)]
pub struct RegexCaptures {
    groups: Vec<Option<RegexMatch>>,
}

impl RegexCaptures {
    pub fn from_standard(captures: &regex::Captures) -> Self {
        let groups = (0..captures.len())
            .map(|index| {
                captures.get(index).map(|matched| RegexMatch {
                    start: matched.start(),
                    end: matched.end(),
                    text: matched.as_str().to_string(),
                })
            })
            .collect();
        Self {
            groups,
        }
    }

    pub fn from_extended(captures: &fancy_regex::Captures) -> Self {
        let groups = (0..captures.len())
            .map(|index| {
                captures.get(index).map(|matched| RegexMatch {
                    start: matched.start(),
                    end: matched.end(),
                    text: matched.as_str().to_string(),
                })
            })
            .collect();
        Self {
            groups,
        }
    }

    pub fn get(
        &self,
        index: usize,
    ) -> Option<&RegexMatch> {
        self.groups.get(index).and_then(|group| group.as_ref())
    }
}
