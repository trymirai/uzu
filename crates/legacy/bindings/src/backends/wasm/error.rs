use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use super::{
    enumeration::string_enumeration_companions,
    typescript::{cfg_attributes, cfg_predicate, custom_sections, js_name, object_member},
};
use crate::contexts::{ErrorContext, ErrorShape};

pub fn error_companions(context: &ErrorContext) -> TokenStream {
    let name = &context.item.ident;
    let value = match context.shape {
        ErrorShape::Unit => string_enumeration_companions(context.item),
        ErrorShape::Data => tagged_error_companions(context.item),
    };
    quote! {
        // A failed call throws or rejects with an `Error` carrying the message, as napi does.
        #[cfg(feature = "bindings-wasm")]
        impl From<#name> for ::wasm_bindgen::JsValue {
            fn from(error: #name) -> Self {
                ::bindings_types::wasm::error(error)
            }
        }

        #value
    }
}

/// An error as a value, e.g. in a stream chunk: napi's `{ type: 'Variant', field: value }`
/// objects, with `field0` for tuple fields.
fn tagged_error_companions(item: &syn::ItemEnum) -> TokenStream {
    let name = &item.ident;
    let type_name = name.to_string();

    let mut into_arms = Vec::new();
    let mut from_arms = Vec::new();
    let mut members = Vec::new();
    for variant in &item.variants {
        let ident = &variant.ident;
        let tag = ident.to_string();
        let cfgs = cfg_attributes(&variant.attrs);
        let fields: Vec<(syn::Ident, String, &syn::Type)> = variant
            .fields
            .iter()
            .enumerate()
            .map(|(index, field)| match &field.ident {
                Some(field_ident) => (field_ident.clone(), js_name(&field_ident.to_string()), &field.ty),
                None => (format_ident!("field{}", index), format!("field{index}"), &field.ty),
            })
            .collect();
        let bindings: Vec<&syn::Ident> = fields.iter().map(|(binding, _, _)| binding).collect();
        let keys: Vec<&String> = fields.iter().map(|(_, key, _)| key).collect();
        let (pattern, constructor) = match &variant.fields {
            syn::Fields::Named(_) => (
                quote! { Self::#ident { #(#bindings),* } },
                quote! { Self::#ident { #(#bindings: ::bindings_types::wasm::FromJs::from_js(::bindings_types::wasm::get_property(&value, #keys)?)?),* } },
            ),
            syn::Fields::Unnamed(_) => (
                quote! { Self::#ident(#(#bindings),*) },
                quote! { Self::#ident(#(::bindings_types::wasm::FromJs::from_js(::bindings_types::wasm::get_property(&value, #keys)?)?),*) },
            ),
            syn::Fields::Unit => (quote! { Self::#ident }, quote! { Self::#ident }),
        };
        into_arms.push(quote! {
            #(#cfgs)*
            #pattern => {
                let object = ::bindings_types::wasm::new_object();
                ::bindings_types::wasm::set_property(&object, "type", ::wasm_bindgen::JsValue::from_str(#tag));
                #(::bindings_types::wasm::set_property(&object, #keys, ::bindings_types::wasm::IntoJs::into_js(#bindings));)*
                object.into()
            }
        });
        from_arms.push(quote! { #(#cfgs)* #tag => ::core::result::Result::Ok(#constructor) });
        let mut member = format!("{{ type: '{tag}'");
        for (_, key, ty) in &fields {
            member.push_str(&format!("; {}", object_member(key, ty)));
        }
        member.push_str(" }");
        members.push(member);
    }

    let predicates: Vec<_> = item.variants.iter().map(|variant| cfg_predicate(&variant.attrs)).collect();
    let declarations = custom_sections(&type_name, &predicates, |included| {
        let members: Vec<String> = members
            .iter()
            .zip(included)
            .filter(|(_, included)| **included)
            .map(|(member, _)| format!("\n  | {member}"))
            .collect();
        format!("export type {type_name} ={};", members.concat())
    });

    quote! {
        #[cfg(feature = "bindings-wasm")]
        impl ::bindings_types::wasm::IntoJs for #name {
            fn into_js(self) -> ::wasm_bindgen::JsValue {
                match self {
                    #(#into_arms),*
                }
            }
        }

        #[cfg(feature = "bindings-wasm")]
        impl ::bindings_types::wasm::FromJs for #name {
            fn from_js(value: ::wasm_bindgen::JsValue) -> ::bindings_types::wasm::JsResult<Self> {
                match ::bindings_types::wasm::get_tag(&value, #type_name)?.as_str() {
                    #(#from_arms,)*
                    other => ::core::result::Result::Err(::bindings_types::wasm::unknown_variant(#type_name, other)),
                }
            }
        }

        #declarations
    }
}
