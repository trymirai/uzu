#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RegexMatch {
    pub start: usize,
    pub end: usize,
    pub text: String,
}
