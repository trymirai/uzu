mod napi;
mod pyo3;
mod rust;
mod uniffi;
mod wasm;

pub use napi::Napi;
use proc_macro2::TokenStream;
pub use pyo3::Pyo3;
use quote::quote;
pub use rust::Rust;
pub use uniffi::Uniffi;
pub use wasm::Wasm;

use crate::contexts::{
    AliasContext, ClassContext, EnumerationContext, ErrorContext, ImplementationContext, MethodMetadata,
    StructureContext,
};

pub trait Backend {
    fn structure_attributes(_context: &StructureContext) -> TokenStream {
        quote! {}
    }

    fn structure_companions(_context: &StructureContext) -> TokenStream {
        quote! {}
    }

    fn class_attributes(_context: &ClassContext) -> TokenStream {
        quote! {}
    }

    fn class_companions(_context: &ClassContext) -> TokenStream {
        quote! {}
    }

    fn enumeration_attributes(_context: &EnumerationContext) -> TokenStream {
        quote! {}
    }

    fn enumeration_companions(_context: &EnumerationContext) -> TokenStream {
        quote! {}
    }

    fn alias_attributes(_context: &AliasContext) -> TokenStream {
        quote! {}
    }

    fn alias_companions(_context: &AliasContext) -> TokenStream {
        quote! {}
    }

    fn implementation_attributes(_context: &ImplementationContext) -> TokenStream {
        quote! {}
    }

    fn implementation_companions(_context: &ImplementationContext) -> TokenStream {
        quote! {}
    }

    fn method_attributes(_metadata: &MethodMetadata) -> TokenStream {
        quote! {}
    }

    fn method_companions(
        _context: &ImplementationContext,
        _metadata: &MethodMetadata,
    ) -> TokenStream {
        quote! {}
    }

    fn error_attributes(_context: &ErrorContext) -> TokenStream {
        quote! {}
    }

    fn error_companions(_context: &ErrorContext) -> TokenStream {
        quote! {}
    }
}
