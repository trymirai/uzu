mod backends;
mod contexts;
mod types;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse_macro_input;

use crate::{
    backends::Backend,
    contexts::{AliasContext, ClassContext, EnumerationContext, ErrorContext, ImplementationContext, StructureContext},
    types::{ClassFlavor, Kind, StructureFlavor},
};

macro_rules! all_backends {
    ($method:ident, $($args:tt)*) => {{
        let rust_tokens = $crate::backends::Rust::$method($($args)*);
        let napi_tokens = $crate::backends::Napi::$method($($args)*);
        let pyo3_tokens = $crate::backends::Pyo3::$method($($args)*);
        let uniffi_tokens = $crate::backends::Uniffi::$method($($args)*);
        let wasm_tokens = $crate::backends::Wasm::$method($($args)*);
        ::quote::quote! {
            #rust_tokens
            #napi_tokens
            #pyo3_tokens
            #uniffi_tokens
            #wasm_tokens
        }
    }};
}

#[proc_macro_attribute]
pub fn export(
    arguments: TokenStream,
    item: TokenStream,
) -> TokenStream {
    let kind = parse_macro_input!(arguments as Kind);

    match kind {
        Kind::Enumeration => dispatch_enumeration(item),
        Kind::Structure(flavor) => dispatch_structure(item, flavor),
        Kind::Class(flavor) => dispatch_class(item, flavor),
        Kind::Alias => dispatch_alias(item),
        Kind::Implementation => dispatch_implementation(item),
        Kind::Method(_) => item,
        Kind::Error => dispatch_error(item),
    }
}

fn dispatch_enumeration(item: TokenStream) -> TokenStream {
    let parsed: syn::ItemEnum = match syn::parse(item) {
        Ok(item_enum) => item_enum,
        Err(error) => return error.to_compile_error().into(),
    };
    let context = match EnumerationContext::from_item(&parsed) {
        Ok(context) => context,
        Err(error) => return error.to_compile_error().into(),
    };
    let attributes = all_backends!(enumeration_attributes, &context);
    let companions = all_backends!(enumeration_companions, &context);
    quote! {
        #attributes
        #parsed
        #companions
    }
    .into()
}

fn dispatch_structure(
    item: TokenStream,
    flavor: StructureFlavor,
) -> TokenStream {
    let parsed: syn::ItemStruct = match syn::parse(item) {
        Ok(item_struct) => item_struct,
        Err(error) => return error.to_compile_error().into(),
    };
    let context = StructureContext {
        flavor,
        item: &parsed,
    };
    let attributes = all_backends!(structure_attributes, &context);
    let companions = all_backends!(structure_companions, &context);
    quote! {
        #attributes
        #parsed
        #companions
    }
    .into()
}

fn dispatch_class(
    item: TokenStream,
    flavor: ClassFlavor,
) -> TokenStream {
    let parsed: syn::ItemStruct = match syn::parse(item) {
        Ok(item_struct) => item_struct,
        Err(error) => return error.to_compile_error().into(),
    };
    let context = ClassContext {
        flavor,
        item: &parsed,
    };
    let attributes = all_backends!(class_attributes, &context);
    let companions = all_backends!(class_companions, &context);
    quote! {
        #attributes
        #parsed
        #companions
    }
    .into()
}

fn dispatch_alias(item: TokenStream) -> TokenStream {
    let parsed: syn::ItemType = match syn::parse(item) {
        Ok(item_type) => item_type,
        Err(error) => return error.to_compile_error().into(),
    };
    let context = AliasContext;
    let attributes = all_backends!(alias_attributes, &context);
    let companions = all_backends!(alias_companions, &context);
    quote! {
        #attributes
        #parsed
        #companions
    }
    .into()
}

fn dispatch_error(item: TokenStream) -> TokenStream {
    let parsed: syn::ItemEnum = match syn::parse(item) {
        Ok(item_enum) => item_enum,
        Err(error) => return error.to_compile_error().into(),
    };
    let context = ErrorContext::from_item(&parsed);
    let attributes = all_backends!(error_attributes, &context);
    let companions = all_backends!(error_companions, &context);
    quote! {
        #attributes
        #parsed
        #companions
    }
    .into()
}

fn dispatch_implementation(item: TokenStream) -> TokenStream {
    let context = match ImplementationContext::from_token_stream(item.into()) {
        Ok(context) => context,
        Err(error) => return error.to_compile_error().into(),
    };

    let attributes = all_backends!(implementation_attributes, &context);
    let companions = all_backends!(implementation_companions, &context);
    let item_implementation = &context.item;

    let method_companions: Vec<TokenStream2> =
        context.all_methods().map(|metadata| all_backends!(method_companions, &context, metadata)).collect();

    quote! {
        #attributes
        #item_implementation
        #companions
        #( #method_companions )*
    }
    .into()
}
