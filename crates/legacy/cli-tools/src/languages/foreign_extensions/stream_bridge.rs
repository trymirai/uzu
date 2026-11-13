/// A type whose `next` yields items until it returns nothing, optionally with a token that cancels the producer.
pub struct StreamBridge {
    pub type_name: String,
    pub next_method_name: String,
    pub item_type: syn::Type,
    pub cancel_getter_name: Option<String>,
}
