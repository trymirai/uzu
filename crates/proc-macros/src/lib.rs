mod dsl;
mod uzu_config;
mod uzu_test;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn kernel(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    dsl::kernel(args, input)
}

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

#[proc_macro_attribute]
pub fn __internal_uzu_test(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    uzu_test::__internal_uzu_test(args, input)
}

#[proc_macro_attribute]
pub fn __internal_uzu_bench(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    uzu_test::__internal_uzu_bench(args, input)
}

#[proc_macro_attribute]
pub fn __internal_uzu_ignored(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    uzu_test::__internal_uzu_ignored(args, input)
}
