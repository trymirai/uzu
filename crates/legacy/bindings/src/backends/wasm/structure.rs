use proc_macro2::TokenStream;
use quote::quote;

use super::{
    fields::{
        accessors, constructor_parameters, field_declarations, field_initializers, named_fields, object_initializers,
        object_members, object_setters, reject_cfg_fields,
    },
    typescript::{cfg_predicate, custom_sections},
};

/// The name of the hidden method that copies a class value out of a JavaScript object, see
/// `bindings_types::wasm::take_clone`.
pub fn clone_method(type_name: &str) -> String {
    format!("__uzuClone{type_name}")
}

/// Annotates the public fields of a wasm class so wasm-bindgen doesn't generate its own
/// snake_case accessors for them; ours are camelCase and convert values like napi.
pub fn skip_public_fields(item: &mut syn::ItemStruct) {
    for field in item.fields.iter_mut() {
        if matches!(field.vis, syn::Visibility::Public(_)) {
            field.attrs.push(syn::parse_quote! { #[cfg_attr(feature = "bindings-wasm", wasm_bindgen(skip))] });
        }
    }
}

/// A data structure exposed as a class: napi's constructor taking every field in order, a getter and
/// setter per field, and conversions that copy the value in and out of JavaScript.
pub fn class_structure_companions(item: &syn::ItemStruct) -> TokenStream {
    let name = &item.ident;
    let type_name = name.to_string();
    let fields = named_fields(&item.fields);
    if let Some(error) = reject_cfg_fields(&fields) {
        return error;
    }
    let parameters = constructor_parameters(&fields, &type_name);
    let initializers = field_initializers(&fields);
    let declarations = field_declarations(&type_name, &[], &fields);
    let accessors = accessors(
        &fields,
        |field| {
            let ident = field.ident;
            quote! { &self.#ident }
        },
        |field| {
            let ident = field.ident;
            quote! { self.#ident = __wasm_value; }
        },
    );
    let clone_method = clone_method(&type_name);

    quote! {
        #[cfg(feature = "bindings-wasm")]
        #[wasm_bindgen::prelude::wasm_bindgen]
        impl #name {
            #[wasm_bindgen(constructor)]
            #[allow(non_snake_case)]
            pub fn __wasm_new(#(#parameters),*) -> ::core::result::Result<#name, ::wasm_bindgen::JsValue> {
                ::core::result::Result::Ok(Self { #(#initializers),* })
            }

            #(#accessors)*

            #[wasm_bindgen(js_name = #clone_method, skip_typescript)]
            pub fn __wasm_clone_handle(&self) -> ::wasm_bindgen::JsValue {
                ::bindings_types::wasm::clone_handle(self)
            }
        }

        #[cfg(feature = "bindings-wasm")]
        impl ::bindings_types::wasm::IntoJs for #name {
            fn into_js(self) -> ::wasm_bindgen::JsValue {
                ::wasm_bindgen::JsValue::from(self)
            }
        }

        #[cfg(feature = "bindings-wasm")]
        impl ::bindings_types::wasm::FromJs for #name {
            fn from_js(value: ::wasm_bindgen::JsValue) -> ::bindings_types::wasm::JsResult<Self> {
                ::bindings_types::wasm::take_clone(&value, #clone_method, #type_name)
            }
        }

        #declarations
    }
}

/// A data structure exposed as a plain object with camelCase keys, as napi's `object`.
pub fn plain_structure_companions(item: &syn::ItemStruct) -> TokenStream {
    let name = &item.ident;
    let type_name = name.to_string();
    let fields = named_fields(&item.fields);
    let setters = object_setters(&fields, |field| {
        let ident = field.ident;
        quote! { self.#ident }
    });
    let initializers = object_initializers(&fields);
    let members = object_members(&fields);
    let predicates: Vec<_> = fields.iter().map(|field| cfg_predicate(field.attributes)).collect();
    let declarations = custom_sections(&type_name, &predicates, |included| {
        let members: Vec<_> = members
            .iter()
            .zip(included)
            .filter(|(_, included)| **included)
            .map(|(member, _)| format!("  {member};\n"))
            .collect();
        format!("export interface {type_name} {{\n{}}}", members.concat())
    });

    quote! {
        #[cfg(feature = "bindings-wasm")]
        impl ::bindings_types::wasm::IntoJs for #name {
            fn into_js(self) -> ::wasm_bindgen::JsValue {
                let object = ::bindings_types::wasm::new_object();
                #(#setters)*
                object.into()
            }
        }

        #[cfg(feature = "bindings-wasm")]
        impl ::bindings_types::wasm::FromJs for #name {
            fn from_js(value: ::wasm_bindgen::JsValue) -> ::bindings_types::wasm::JsResult<Self> {
                ::bindings_types::wasm::expect_object(&value, #type_name)?;
                ::core::result::Result::Ok(Self { #(#initializers),* })
            }
        }

        #declarations
    }
}

/// A class wrapping Rust state (an engine, a session): returned to JavaScript as is.
pub fn class_companions(item: &syn::ItemStruct) -> TokenStream {
    let name = &item.ident;
    quote! {
        #[cfg(feature = "bindings-wasm")]
        impl ::bindings_types::wasm::IntoJs for #name {
            fn into_js(self) -> ::wasm_bindgen::JsValue {
                ::wasm_bindgen::JsValue::from(self)
            }
        }
    }
}
