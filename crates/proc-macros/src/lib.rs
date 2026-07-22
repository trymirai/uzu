mod dsl;
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

/// Marks an enum as the Rust-side sum type for a group of shader template axes, named
/// in declaration order: `#[variant_group(B_PROLOGUE, BITS, GROUP_SIZE)]`.
///
/// The backend-uzu build script enumerates the enum's legal field combinations instead
/// of the raw cross-product of those axes, so combinations the type cannot represent are
/// never instantiated. Expands to the item unchanged.
#[proc_macro_attribute]
pub fn variant_group(
    _args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    input
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
