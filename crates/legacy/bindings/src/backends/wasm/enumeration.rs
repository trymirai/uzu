use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use super::{
    fields::{
        accessors, constructor_parameters, field_declarations, field_initializers, named_fields, reject_cfg_fields,
    },
    structure::clone_method,
    typescript::{cfg_attributes, cfg_predicate, custom_sections},
};

/// An enum of unit variants as napi's string enum: each value is the variant name, the d.ts
/// declares an enum, and a runtime object maps the variant names to their values. Not a `const enum`:
/// TypeScript can't inline one under `isolatedModules`, see the napi package's generate-napi-wrappers.cjs.
pub fn string_enumeration_companions(item: &syn::ItemEnum) -> TokenStream {
    let name = &item.ident;
    let type_name = name.to_string();
    let object = format_ident!("__WasmEnum{}", name);

    let mut into_arms = Vec::new();
    let mut from_arms = Vec::new();
    let mut getters = Vec::new();
    for variant in &item.variants {
        let ident = &variant.ident;
        let value = ident.to_string();
        let cfgs = cfg_attributes(&variant.attrs);
        let pattern = match variant.fields {
            syn::Fields::Unit => quote! { Self::#ident },
            _ => quote! { Self::#ident {} },
        };
        into_arms.push(quote! { #(#cfgs)* #pattern => ::wasm_bindgen::JsValue::from_str(#value) });
        from_arms.push(quote! { #(#cfgs)* #value => ::core::result::Result::Ok(#pattern) });
        let getter = format_ident!("__wasm_variant_{}", ident);
        getters.push(quote! {
            #[cfg(feature = "bindings-wasm")]
            #(#cfgs)*
            #[wasm_bindgen::prelude::wasm_bindgen(js_class = #type_name)]
            impl #object {
                #[wasm_bindgen(getter = #value, skip_typescript)]
                #[allow(non_snake_case)]
                pub fn #getter() -> ::std::string::String {
                    ::std::string::String::from(#value)
                }
            }
        });
    }

    let values: Vec<String> = item.variants.iter().map(|variant| variant.ident.to_string()).collect();
    let predicates: Vec<_> = item.variants.iter().map(|variant| cfg_predicate(&variant.attrs)).collect();
    let declarations = custom_sections(&type_name, &predicates, |included| {
        let members: Vec<String> = values
            .iter()
            .zip(included)
            .filter(|(_, included)| **included)
            .map(|(value, _)| format!("  {value} = '{value}',\n"))
            .collect();
        format!("export declare enum {type_name} {{\n{}}}", members.concat())
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
                let Some(value) = value.as_string() else {
                    return ::core::result::Result::Err(::bindings_types::wasm::type_error(concat!("expected ", #type_name)));
                };
                match value.as_str() {
                    #(#from_arms,)*
                    other => ::core::result::Result::Err(::bindings_types::wasm::unknown_variant(#type_name, other)),
                }
            }
        }

        #[cfg(feature = "bindings-wasm")]
        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        #[wasm_bindgen::prelude::wasm_bindgen(js_name = #type_name, skip_typescript)]
        pub struct #object {}

        #(#getters)*

        #declarations
    }
}

/// An enum with data as napi's variant classes: a class per variant named `{Enum}{Variant}` with
/// its fields, and a union type of the variant classes under the enum's name.
pub fn data_enumeration_companions(item: &syn::ItemEnum) -> TokenStream {
    let name = &item.ident;
    let type_name = name.to_string();
    let clone_method = clone_method(&type_name);

    let mut classes = Vec::new();
    let mut into_arms = Vec::new();
    for variant in &item.variants {
        let variant_ident = &variant.ident;
        let class_name = format!("{type_name}{variant_ident}");
        let wrapper = format_ident!("__Wasm{}{}", name, variant_ident);
        let cfgs = cfg_attributes(&variant.attrs);
        let fields = named_fields(&variant.fields);
        if let Some(error) = reject_cfg_fields(&fields) {
            return error;
        }
        let parameters = constructor_parameters(&fields, &class_name);
        let initializers = field_initializers(&fields);
        let declarations = field_declarations(&class_name, &cfgs, &fields);
        let accessors = accessors(
            &fields,
            |field| {
                let ident = field.ident;
                quote! {
                    match &self.inner {
                        #name::#variant_ident { #ident, .. } => #ident,
                        _ => ::core::unreachable!(),
                    }
                }
            },
            |field| {
                let ident = field.ident;
                quote! {
                    if let #name::#variant_ident { #ident, .. } = &mut self.inner {
                        *#ident = __wasm_value;
                    }
                }
            },
        );

        classes.push(quote! {
            #[cfg(feature = "bindings-wasm")]
            #(#cfgs)*
            #[doc(hidden)]
            #[allow(non_camel_case_types)]
            #[wasm_bindgen::prelude::wasm_bindgen(js_name = #class_name)]
            pub struct #wrapper {
                inner: #name,
            }

            #[cfg(feature = "bindings-wasm")]
            #(#cfgs)*
            #[allow(unreachable_patterns, irrefutable_let_patterns)]
            #[wasm_bindgen::prelude::wasm_bindgen(js_class = #class_name)]
            impl #wrapper {
                #[wasm_bindgen(constructor)]
                #[allow(non_snake_case)]
                pub fn __wasm_new(#(#parameters),*) -> ::core::result::Result<#wrapper, ::wasm_bindgen::JsValue> {
                    ::core::result::Result::Ok(Self { inner: #name::#variant_ident { #(#initializers),* } })
                }

                #(#accessors)*

                #[wasm_bindgen(js_name = #clone_method, skip_typescript)]
                pub fn __wasm_clone_pointer(&self) -> usize {
                    ::bindings_types::wasm::clone_pointer(&self.inner)
                }
            }

            #declarations
        });
        into_arms.push(quote! {
            #(#cfgs)*
            value @ Self::#variant_ident { .. } => ::wasm_bindgen::JsValue::from(#wrapper { inner: value })
        });
    }

    let class_names: Vec<String> =
        item.variants.iter().map(|variant| format!("{type_name}{}", variant.ident)).collect();
    let predicates: Vec<_> = item.variants.iter().map(|variant| cfg_predicate(&variant.attrs)).collect();
    let declarations = custom_sections(&type_name, &predicates, |included| {
        let members: Vec<&str> = class_names
            .iter()
            .zip(included)
            .filter(|(_, included)| **included)
            .map(|(class_name, _)| class_name.as_str())
            .collect();
        let union = if members.is_empty() {
            "never".to_string()
        } else {
            members.join(" | ")
        };
        format!("export type {type_name} = {union};")
    });

    quote! {
        #(#classes)*

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
                ::bindings_types::wasm::take_clone(&value, #clone_method, #type_name)
            }
        }

        #declarations
    }
}
