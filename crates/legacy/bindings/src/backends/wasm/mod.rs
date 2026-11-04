//! wasm-bindgen bindings with the same JavaScript API as the napi ones.
//!
//! Values cross the boundary through `bindings_types::wasm::{IntoJs, FromJs}` rather than through
//! wasm-bindgen's own conversions or serde, so the shapes match napi (camelCase fields, a class per
//! enum variant, string enums, `null` for a missing value) whatever the types' serde attributes say.
//! Exported structures and classes are wasm-bindgen classes; every exported method gets a wrapper
//! that converts its arguments and result, and TypeScript the wrapper can't express (unions, enums,
//! interfaces) is emitted as custom d.ts sections.

mod enumeration;
mod error;
mod fields;
mod method;
mod structure;
mod typescript;

use proc_macro2::TokenStream;
use quote::quote;

use crate::{
    backends::Backend,
    contexts::{
        AliasContext, ClassContext, EnumerationContext, EnumerationShape, ErrorContext, ImplementationContext,
        MethodMetadata, StructureContext,
    },
    types::StructureFlavor,
};

pub struct Wasm;

impl Backend for Wasm {
    fn prepare_structure(
        item: &mut syn::ItemStruct,
        flavor: &StructureFlavor,
    ) {
        if matches!(flavor, StructureFlavor::Class) {
            structure::skip_public_fields(item);
        }
    }

    fn prepare_class(item: &mut syn::ItemStruct) {
        structure::skip_public_fields(item);
    }

    fn structure_attributes(context: &StructureContext) -> TokenStream {
        match context.flavor {
            StructureFlavor::Plain => quote! {},
            StructureFlavor::Class => quote! {
                #[cfg_attr(feature = "bindings-wasm", wasm_bindgen::prelude::wasm_bindgen)]
            },
        }
    }

    fn structure_companions(context: &StructureContext) -> TokenStream {
        match context.flavor {
            StructureFlavor::Plain => structure::plain_structure_companions(context.item),
            StructureFlavor::Class => structure::class_structure_companions(context.item),
        }
    }

    fn class_attributes(_context: &ClassContext) -> TokenStream {
        quote! {
            #[cfg_attr(feature = "bindings-wasm", wasm_bindgen::prelude::wasm_bindgen)]
        }
    }

    fn class_companions(context: &ClassContext) -> TokenStream {
        structure::class_companions(context.item)
    }

    fn enumeration_companions(context: &EnumerationContext) -> TokenStream {
        match context.shape {
            EnumerationShape::Unit => enumeration::string_enumeration_companions(context.item),
            EnumerationShape::Data => enumeration::data_enumeration_companions(context.item),
        }
    }

    fn alias_companions(context: &AliasContext) -> TokenStream {
        let name = context.item.ident.to_string();
        let declaration = format!("export type {name} = {};", typescript::typescript_type(&context.item.ty, &name));
        typescript::custom_section(&name, &typescript::cfg_attributes(&context.item.attrs), &declaration)
    }

    fn error_companions(context: &ErrorContext) -> TokenStream {
        error::error_companions(context)
    }

    fn method_companions(
        context: &ImplementationContext,
        metadata: &MethodMetadata,
    ) -> TokenStream {
        method::method_companions(context, metadata)
    }
}
