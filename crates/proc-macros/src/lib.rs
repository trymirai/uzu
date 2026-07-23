mod dsl;
mod tool;
mod uzu_config;
mod uzu_test;

use proc_macro::TokenStream;

// CPU kernel DSL
#[proc_macro_attribute]
pub fn kernel(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    dsl::kernel(args, input)
}

// Config DSL
#[proc_macro_attribute]
pub fn uzu_config_abstract(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    uzu_config::uzu_config_abstract(args, input)
}

#[proc_macro_attribute]
pub fn uzu_config(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    uzu_config::uzu_config(args, input)
}

// Test DSL
#[proc_macro_attribute]
pub fn uzu_test(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    uzu_test::uzu_test(args, input)
}

#[proc_macro_attribute]
pub fn uzu_bench(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    uzu_test::uzu_bench(args, input)
}

// Tool calls
#[proc_macro_attribute]
pub fn uzu_tool_function(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    tool::tool_function(args, input)
}

#[proc_macro_derive(UzuToolSchema, attributes(serde))]
pub fn uzu_tool_schema(input: TokenStream) -> TokenStream {
    tool::derive_tool_schema(input)
}
