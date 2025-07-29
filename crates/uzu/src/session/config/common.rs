use serde::Deserialize;
use tokenizers::AddedToken;

#[derive(Clone, Deserialize, Debug)]
#[serde(untagged)]
pub enum ValueOrList<T: Clone> {
    Value(T),
    List(Vec<T>),
}

impl<T: Clone> ValueOrList<T> {
    pub fn to_list(&self) -> Vec<T> {
        match self {
            ValueOrList::Value(value) => vec![value.clone()],
            ValueOrList::List(list) => list.clone(),
        }
    }
}

#[derive(Clone, Deserialize, Debug)]
#[serde(untagged)]
pub enum ValueOrToken {
    Value(String),
    Token(AddedToken),
    Property {
        content: String,
    },
}

impl ValueOrToken {
    pub fn to_string(&self) -> String {
        match self {
            ValueOrToken::Value(value) => value.clone(),
            ValueOrToken::Token(token) => token.content.clone(),
            ValueOrToken::Property {
                content,
            } => content.clone(),
        }
    }
}
