use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use super::typescript::{cfg_attributes, custom_section, doc_attributes, js_name};
use crate::{
    backends::typescript::{is_unit, option_inner, result_inner, typescript_type},
    contexts::{ImplementationContext, MethodMetadata},
    types::MethodFlavor,
};

/// Static getter that marks a stream class, so the package entry can give it `Symbol.asyncIterator`.
const STREAM_MARKER: &str = "__uzuStream";

pub fn method_companions(
    context: &ImplementationContext,
    metadata: &MethodMetadata,
) -> TokenStream {
    let self_name = context.self_type_ident().map(ToString::to_string).unwrap_or_default();
    match metadata.flavor {
        MethodFlavor::FactoryWithCallback => callback_factory(&context.self_type, metadata),
        MethodFlavor::StreamNext => {
            let wrapper = wrapper(&context.self_type, &self_name, metadata);
            let stream = stream_declarations(&context.self_type, &self_name, metadata);
            quote! { #wrapper #stream }
        },
        _ => wrapper(&context.self_type, &self_name, metadata),
    }
}

/// A `#[wasm_bindgen]` method calling the exported one, with napi's JavaScript name. Arguments arrive
/// as JavaScript values and go through `FromJs`, except `&Class` ones, which wasm-bindgen borrows;
/// the result goes through `IntoJs`. Each wrapper has its own impl block carrying the method's cfgs.
fn wrapper(
    self_type: &syn::Type,
    self_name: &str,
    metadata: &MethodMetadata,
) -> TokenStream {
    let method = &metadata.method;
    let name = &method.sig.ident;
    let wrapper_name = format_ident!("__wasm_{}", name);
    let cfgs = cfg_attributes(&method.attrs);
    let docs = doc_attributes(&method.attrs);
    let asyncness = &method.sig.asyncness;
    let await_call = metadata.is_async().then(|| quote! { .await });
    let receiver = metadata.receiver().map(|receiver| quote! { #receiver, });

    let arguments = metadata.typed_args();
    let optional_from =
        arguments.iter().rposition(|argument| option_inner(&argument.ty).is_none()).map_or(0, |position| position + 1);
    let mut parameters = Vec::new();
    let mut conversions = Vec::new();
    let mut call_arguments = Vec::new();
    for (index, argument) in arguments.iter().enumerate() {
        let syn::Pat::Ident(pattern) = argument.pat.as_ref() else {
            return syn::Error::new_spanned(&argument.pat, "bindings::export expects plain argument names")
                .to_compile_error();
        };
        let parameter = format_ident!("{}", js_name(&pattern.ident.to_string()));
        let ty = argument.ty.as_ref();
        if let syn::Type::Reference(reference) = ty {
            // An exported class, borrowed from the JavaScript object without copying it.
            parameters.push(quote! { #parameter: #reference });
            call_arguments.push(quote! { #parameter });
            continue;
        }
        let declaration = match option_inner(ty) {
            Some(inner) if index >= optional_from => {
                let ty = format!("{} | null", typescript_type(inner, self_name));
                quote! { #[wasm_bindgen(unchecked_optional_param_type = #ty)] }
            },
            Some(inner) => {
                let ty = format!("{} | undefined | null", typescript_type(inner, self_name));
                quote! { #[wasm_bindgen(unchecked_param_type = #ty)] }
            },
            None => {
                let ty = typescript_type(ty, self_name);
                quote! { #[wasm_bindgen(unchecked_param_type = #ty)] }
            },
        };
        parameters.push(quote! { #declaration #parameter: ::wasm_bindgen::JsValue });
        conversions.push(quote! { let #parameter = <#ty as ::bindings_types::wasm::FromJs>::from_js(#parameter)?; });
        call_arguments.push(quote! { #parameter });
    }

    let call = if metadata.receiver().is_some() {
        quote! { self.#name(#(#call_arguments),*) #await_call }
    } else {
        quote! { Self::#name(#(#call_arguments),*) #await_call }
    };
    let unit: syn::Type = syn::parse_quote! { () };
    let output = match &method.sig.output {
        syn::ReturnType::Default => &unit,
        syn::ReturnType::Type(_, ty) => ty.as_ref(),
    };
    let (value_type, call) = match result_inner(output) {
        Some(inner) => (inner, quote! { #call.map_err(::bindings_types::wasm::error)? }),
        None => (output, call),
    };

    let js_name = match metadata.flavor {
        // napi keeps the Rust name of factories.
        MethodFlavor::Factory => name.to_string(),
        MethodFlavor::StreamNext => "next".to_string(),
        MethodFlavor::Setter => js_name(name.to_string().trim_start_matches("set_")),
        _ => js_name(&name.to_string()),
    };
    let kind = match metadata.flavor {
        MethodFlavor::Constructor => quote! { constructor },
        MethodFlavor::Getter => quote! { getter = #js_name },
        MethodFlavor::Setter => quote! { setter = #js_name },
        _ => quote! { js_name = #js_name },
    };

    let (return_type, attributes, body) = if matches!(metadata.flavor, MethodFlavor::Constructor) {
        (quote! { ::core::result::Result<Self, ::wasm_bindgen::JsValue> }, quote! { #kind }, quote! { Ok(#call) })
    } else if is_unit(value_type) {
        (
            quote! { ::core::result::Result<(), ::wasm_bindgen::JsValue> },
            quote! { #kind },
            quote! {
                #call;
                Ok(())
            },
        )
    } else {
        let returned = typescript_type(value_type, self_name);
        (
            quote! { ::core::result::Result<::wasm_bindgen::JsValue, ::wasm_bindgen::JsValue> },
            quote! { #kind, unchecked_return_type = #returned },
            quote! { Ok(::bindings_types::wasm::IntoJs::into_js(#call)) },
        )
    };

    quote! {
        #[cfg(feature = "bindings-wasm")]
        #(#cfgs)*
        #[wasm_bindgen::prelude::wasm_bindgen]
        impl #self_type {
            #(#docs)*
            #[wasm_bindgen(#attributes)]
            #[allow(non_snake_case, clippy::unused_unit)]
            pub #asyncness fn #wrapper_name(#receiver #(#parameters),*) -> #return_type {
                #(#conversions)*
                #body
            }
        }
    }
}

/// napi's `[Symbol.asyncIterator]` on stream classes: declared here and installed at runtime by the
/// package entry on every class carrying the stream marker.
fn stream_declarations(
    self_type: &syn::Type,
    self_name: &str,
    metadata: &MethodMetadata,
) -> TokenStream {
    let cfgs = cfg_attributes(&metadata.method.attrs);
    let item = match &metadata.method.sig.output {
        syn::ReturnType::Type(_, ty) => option_inner(ty).map(|inner| typescript_type(inner, self_name)),
        syn::ReturnType::Default => None,
    };
    let Some(item) = item else {
        return syn::Error::new_spanned(&metadata.method.sig, "Method(StreamNext) must return `Option<T>`")
            .to_compile_error();
    };
    let declaration = format!(
        "export interface {self_name} {{\n  [Symbol.asyncIterator](): AsyncGenerator<{item}, void, undefined>\n}}"
    );
    let section = custom_section(&format!("{self_name}AsyncIterator"), &cfgs, &declaration);
    quote! {
        #section

        #[cfg(feature = "bindings-wasm")]
        #(#cfgs)*
        #[wasm_bindgen::prelude::wasm_bindgen]
        impl #self_type {
            #[wasm_bindgen(getter = #STREAM_MARKER, skip_typescript)]
            pub fn __wasm_stream_marker() -> bool {
                true
            }
        }
    }
}

/// A static factory taking a JavaScript function, called without waiting for it.
fn callback_factory(
    self_type: &syn::Type,
    metadata: &MethodMetadata,
) -> TokenStream {
    let Some(inputs) = metadata.callback_inputs() else {
        return syn::Error::new_spanned(&metadata.method.sig, "expected a boxed void callback").to_compile_error();
    };
    let self_name = match self_type {
        syn::Type::Path(path) => path.path.segments.last().map(|segment| segment.ident.to_string()),
        _ => None,
    }
    .unwrap_or_default();
    let arguments: Vec<_> = (0..inputs.len()).map(|index| format_ident!("arg{index}")).collect();
    let signature = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| format!("arg{index}: {}", typescript_type(input, &self_name)))
        .collect::<Vec<_>>()
        .join(", ");
    let callback_type = format!("({signature}) => void");
    let name = &metadata.method.sig.ident;
    let js_name = js_name(&name.to_string());
    let cfgs = cfg_attributes(&metadata.method.attrs);
    let body = &metadata.method.block;
    quote! {
        #[cfg(feature = "bindings-wasm")]
        #(#cfgs)*
        #[wasm_bindgen::prelude::wasm_bindgen]
        impl #self_type {
            #[wasm_bindgen(js_name = #js_name)]
            pub fn #name(#[wasm_bindgen(unchecked_param_type = #callback_type)] callback: ::bindings_types::js_sys::Function) -> Self {
                let callback: Box<dyn Fn(#(#inputs),*) + Send + Sync> = Box::new(move |#(#arguments: #inputs),*| {
                    let arguments = ::bindings_types::js_sys::Array::new();
                    #(arguments.push(&::bindings_types::wasm::IntoJs::into_js(#arguments));)*
                    let _ = callback.apply(&::wasm_bindgen::JsValue::UNDEFINED, &arguments);
                });
                #body
            }
        }
    }
}
