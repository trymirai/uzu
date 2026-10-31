/// A factory taking a Rust closure, which UniFFI exports as a `<Type>Handler` callback interface with `on_event`.
pub struct CallbackFactory {
    pub type_name: String,
    pub method_name: String,
    pub callback_arg_types: Vec<syn::Type>,
}
