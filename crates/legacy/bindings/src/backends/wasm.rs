use heck::ToLowerCamelCase;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::{
    backends::Backend,
    contexts::{
        ClassContext, EnumerationContext, ErrorContext, ImplementationContext, MethodMetadata, StructureContext,
    },
    types::{MethodFlavor, StructureFlavor},
};

pub struct Wasm;

impl Backend for Wasm {
    fn structure_attributes(context: &StructureContext) -> TokenStream {
        match context.flavor {
            StructureFlavor::Plain => quote! {
                #[cfg_attr(feature = "bindings-wasm", derive(tsify::Tsify))]
                #[cfg_attr(feature = "bindings-wasm", tsify(into_wasm_abi, from_wasm_abi))]
            },
            StructureFlavor::Class => quote! {
                #[cfg_attr(feature = "bindings-wasm", wasm_bindgen::prelude::wasm_bindgen(getter_with_clone))]
            },
        }
    }

    fn class_attributes(_context: &ClassContext) -> TokenStream {
        quote! {
            #[cfg_attr(feature = "bindings-wasm", wasm_bindgen::prelude::wasm_bindgen)]
        }
    }

    fn enumeration_attributes(_context: &EnumerationContext) -> TokenStream {
        quote! {
            #[cfg_attr(feature = "bindings-wasm", derive(tsify::Tsify))]
            #[cfg_attr(feature = "bindings-wasm", tsify(into_wasm_abi, from_wasm_abi))]
        }
    }

    fn implementation_attributes(_context: &ImplementationContext) -> TokenStream {
        quote! {
            #[cfg_attr(feature = "bindings-wasm", wasm_bindgen::prelude::wasm_bindgen)]
        }
    }

    fn error_companions(context: &ErrorContext) -> TokenStream {
        let type_name = &context.item.ident;
        quote! {
            #[cfg(feature = "bindings-wasm")]
            impl From<#type_name> for wasm_bindgen::JsValue {
                fn from(error: #type_name) -> Self {
                    wasm_bindgen::JsValue::from_str(&error.to_string())
                }
            }
        }
    }

    fn method_attributes(metadata: &MethodMetadata) -> TokenStream {
        if returns_map(metadata) {
            return quote! { #[cfg(not(feature = "bindings-wasm"))] };
        }
        let name = metadata.method.sig.ident.to_string().to_lower_camel_case();
        let getter = matches!(metadata.flavor, MethodFlavor::Getter).then(|| quote! { getter, });
        quote! {
            #[cfg_attr(feature = "bindings-wasm", wasm_bindgen::prelude::wasm_bindgen(#getter js_name = #name))]
        }
    }

    fn method_companions(
        context: &ImplementationContext,
        metadata: &MethodMetadata,
    ) -> TokenStream {
        if matches!(metadata.flavor, MethodFlavor::FactoryWithCallback) {
            return callback_factory(&context.self_type, metadata);
        }
        if returns_map(metadata) {
            return map_method(&context.self_type, metadata);
        }
        if !matches!(metadata.flavor, MethodFlavor::Factory) {
            return quote! {};
        }
        let self_type = &context.self_type;
        let method_name = &metadata.method.sig.ident;
        let mut method = metadata.method.clone();
        method.sig.ident = format_ident!("__wasm_{}", method_name);
        let args = metadata.arg_idents();
        let await_call = metadata.is_async().then(|| quote! { .await });
        method.block = syn::parse_quote! { { Self::#method_name(#(#args),*) #await_call } };
        let attributes = Self::method_attributes(metadata);
        quote! {
            #[cfg(feature = "bindings-wasm")]
            #[wasm_bindgen::prelude::wasm_bindgen]
            impl #self_type {
                #attributes
                #method
            }
        }
    }
}

fn returns_map(metadata: &MethodMetadata) -> bool {
    let syn::ReturnType::Type(_, ty) = &metadata.method.sig.output else {
        return false;
    };
    let syn::Type::Path(path) = ty.as_ref() else {
        return false;
    };
    path.path.segments.last().is_some_and(|segment| segment.ident == "HashMap")
}

fn map_method(
    self_type: &syn::Type,
    metadata: &MethodMetadata,
) -> TokenStream {
    let mut original = metadata.method.clone();
    original.attrs.clear();
    let name = &original.sig.ident;
    let js_name = name.to_string().to_lower_camel_case();
    let getter = matches!(metadata.flavor, MethodFlavor::Getter).then(|| quote! { getter, });
    let args = metadata.arg_idents();
    let await_call = metadata.is_async().then(|| quote! { .await });
    let call = if metadata.receiver().is_some() {
        quote! { self.#name(#(#args),*) }
    } else {
        quote! { Self::#name(#(#args),*) }
    };
    let mut wrapper = original.clone();
    wrapper.sig.ident = format_ident!("__wasm_{}", name);
    wrapper.sig.output = syn::parse_quote! { -> bindings_types::js_sys::Map };
    wrapper.block = syn::parse_quote! {{
        let entries = #call #await_call;
        let map = bindings_types::js_sys::Map::new();
        for (key, value) in entries {
            map.set(&wasm_bindgen::JsValue::from(key), &wasm_bindgen::JsValue::from(value));
        }
        map
    }};
    quote! {
        #[cfg(feature = "bindings-wasm")]
        impl #self_type { #original }
        #[cfg(feature = "bindings-wasm")]
        #[wasm_bindgen::prelude::wasm_bindgen]
        impl #self_type {
            #[wasm_bindgen::prelude::wasm_bindgen(#getter js_name = #js_name)]
            #wrapper
        }
    }
}

fn callback_factory(
    self_type: &syn::Type,
    metadata: &MethodMetadata,
) -> TokenStream {
    let Some(inputs) = metadata.callback_inputs() else {
        return syn::Error::new_spanned(&metadata.method.sig, "expected a boxed void callback").to_compile_error();
    };
    let args: Vec<_> = (0..inputs.len()).map(|index| format_ident!("arg{index}")).collect();
    let name = &metadata.method.sig.ident;
    let js_name = name.to_string().to_lower_camel_case();
    let body = &metadata.method.block;
    quote! {
        #[cfg(feature = "bindings-wasm")]
        #[wasm_bindgen::prelude::wasm_bindgen]
        impl #self_type {
            #[wasm_bindgen::prelude::wasm_bindgen(js_name = #js_name)]
            pub fn #name(callback: bindings_types::js_sys::Function) -> Self {
                let callback: Box<dyn Fn(#(#inputs),*) + Send + Sync> = Box::new(move |#(#args: #inputs),*| {
                    let arguments = bindings_types::js_sys::Array::new();
                    #(arguments.push(&wasm_bindgen::JsValue::from(#args));)*
                    let _ = callback.apply(&wasm_bindgen::JsValue::UNDEFINED, &arguments);
                });
                #body
            }
        }
    }
}
