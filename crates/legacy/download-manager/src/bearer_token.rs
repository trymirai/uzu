use std::fmt::{Debug, Formatter, Result as FmtResult};

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BearerToken(String);

impl BearerToken {
    pub fn header_value(&self) -> String {
        format!("Bearer {}", self.0)
    }
}

impl From<String> for BearerToken {
    fn from(token: String) -> Self {
        Self(token)
    }
}

impl Debug for BearerToken {
    fn fmt(
        &self,
        formatter: &mut Formatter<'_>,
    ) -> FmtResult {
        formatter.write_str("BearerToken(<redacted>)")
    }
}
