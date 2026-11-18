use heck::{AsLowerCamelCase, AsSnakeCase};

use super::factory_argument::FactoryArgument;

/// A static constructor UniFFI exports as a free function, like `ChatMessage::system` as `chat_message_system`.
pub struct Factory {
    pub type_name: String,
    pub method_name: String,
    pub is_async: bool,
    pub throws: bool,
    pub optional: bool,
    pub arguments: Vec<FactoryArgument>,
}

impl Factory {
    /// The free function's name in Swift and Kotlin, like `chatMessageSystem`.
    pub fn free_function_name(&self) -> String {
        let method = AsLowerCamelCase(&self.method_name).to_string();
        let mut characters = method.chars();
        let capitalized_method = match characters.next() {
            Some(first) => first.to_ascii_uppercase().to_string() + characters.as_str(),
            None => String::new(),
        };
        format!("{}{capitalized_method}", AsLowerCamelCase(AsSnakeCase(&self.type_name).to_string()))
    }
}
