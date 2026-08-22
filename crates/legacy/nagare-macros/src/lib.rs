mod uzu_tool;

use proc_macro::TokenStream;

// Tool calls
#[proc_macro]
pub fn uzu_tool_closure(input: TokenStream) -> TokenStream {
    uzu_tool::uzu_tool_closure(input)
}

#[proc_macro_attribute]
pub fn uzu_tool_function(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    uzu_tool::uzu_tool_function(args, input)
}
