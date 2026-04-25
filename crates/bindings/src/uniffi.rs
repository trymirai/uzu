use quote::{format_ident, quote};
use syn::{Attribute, Ident, parse_quote};

use crate::{BindingKind, MethodFlavor};

pub fn attributes(kind: &BindingKind) -> proc_macro2::TokenStream {
    match kind {
        BindingKind::Enum => quote! {
            #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Enum))]
        },
        BindingKind::Struct => quote! {
            #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Record))]
        },
        BindingKind::Class | BindingKind::Stream => quote! {
            #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Object))]
        },
        BindingKind::ClassCloneable => quote! {
            #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Record))]
        },
        BindingKind::Implementation => quote! {},
        _ => quote! {},
    }
}

pub fn implementation_attribute(has_async: bool) -> proc_macro2::TokenStream {
    if has_async {
        quote! {
            #[cfg_attr(feature = "bindings-uniffi", uniffi::export(async_runtime = "tokio"))]
        }
    } else {
        quote! {
            #[cfg_attr(feature = "bindings-uniffi", uniffi::export)]
        }
    }
}

pub fn method_attribute(flavor: &MethodFlavor) -> Option<Attribute> {
    match flavor {
        MethodFlavor::Plain | MethodFlavor::Getter | MethodFlavor::Setter | MethodFlavor::Factory => None,
        MethodFlavor::Constructor => Some(parse_quote! {
            #[cfg_attr(feature = "bindings-uniffi", uniffi::constructor)]
        }),
    }
}

pub fn factory(
    self_type: &syn::Type,
    method: &syn::ImplItemFn,
) -> proc_macro2::TokenStream {
    let method_ident = &method.sig.ident;
    let self_ident = match self_type_ident(self_type) {
        Some(ident) => ident,
        None => {
            return syn::Error::new_spanned(self_type, "bindings::export(Factory) requires a named self type")
                .to_compile_error();
        },
    };
    let fn_name = format_ident!("{}_{}", heck::AsSnakeCase(self_ident.to_string()).to_string(), method_ident,);
    let inputs = &method.sig.inputs;
    let output = replace_self_in_return(&method.sig.output, self_type);
    let asyncness = &method.sig.asyncness;
    let arg_idents = factory_arg_idents(inputs);
    let forward_await = asyncness.as_ref().map(|_| quote! { .await });
    let export_attr = if asyncness.is_some() {
        quote! { #[uniffi::export(async_runtime = "tokio")] }
    } else {
        quote! { #[uniffi::export] }
    };

    quote! {
        #[cfg(feature = "bindings-uniffi")]
        #export_attr
        pub #asyncness fn #fn_name( #inputs ) #output {
            <#self_type>::#method_ident( #( #arg_idents ),* )#forward_await
        }
    }
}

fn replace_self_in_return(
    output: &syn::ReturnType,
    self_type: &syn::Type,
) -> syn::ReturnType {
    use syn::fold::Fold;
    let mut folder = SelfReplacer {
        self_type: self_type.clone(),
    };
    folder.fold_return_type(output.clone())
}

struct SelfReplacer {
    self_type: syn::Type,
}

impl syn::fold::Fold for SelfReplacer {
    fn fold_type(
        &mut self,
        ty: syn::Type,
    ) -> syn::Type {
        if let syn::Type::Path(path) = &ty {
            if path.qself.is_none()
                && path.path.segments.len() == 1
                && path.path.segments[0].ident == "Self"
                && path.path.segments[0].arguments.is_empty()
            {
                return self.self_type.clone();
            }
        }
        syn::fold::fold_type(self, ty)
    }
}

fn factory_arg_idents(inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>) -> Vec<syn::Ident> {
    inputs
        .iter()
        .filter_map(|arg| match arg {
            syn::FnArg::Typed(pat_type) => match pat_type.pat.as_ref() {
                syn::Pat::Ident(pat_ident) => Some(pat_ident.ident.clone()),
                _ => None,
            },
            syn::FnArg::Receiver(_) => None,
        })
        .collect()
}

pub fn error_attribute() -> proc_macro2::TokenStream {
    quote! {
        #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Error))]
    }
}

#[allow(dead_code)]
pub fn error_implementations(_type_name: &Ident) -> proc_macro2::TokenStream {
    quote! {}
}

pub fn factory_with_callback(
    self_type: &syn::Type,
    method: &syn::ImplItemFn,
) -> proc_macro2::TokenStream {
    let method_ident = &method.sig.ident;
    let body = &method.block;

    let self_ident = match self_type_ident(self_type) {
        Some(ident) => ident,
        None => {
            return syn::Error::new_spanned(
                self_type,
                "bindings::export(FactoryWithCallback) requires a named self type",
            )
            .to_compile_error();
        },
    };
    let handler_ident = format_ident!("{}Handler", self_ident);
    let callback_inputs = match crate::napi::extract_callback_inputs(method) {
        Some(inputs) => inputs,
        None => {
            return syn::Error::new_spanned(
                &method.sig,
                "bindings::export(FactoryWithCallback) requires a parameter of type \
                 `Box<dyn Fn(..) + Send + Sync>` (return type must be `()`)",
            )
            .to_compile_error();
        },
    };
    let arg_idents: Vec<syn::Ident> = (0..callback_inputs.len()).map(|index| format_ident!("arg{index}")).collect();

    quote! {
        #[cfg(feature = "bindings-uniffi")]
        #[uniffi::export(callback_interface)]
        pub trait #handler_ident: Send + Sync {
            fn on_event(&self, #( #arg_idents: #callback_inputs ),*);
        }

        #[cfg(feature = "bindings-uniffi")]
        #[uniffi::export]
        impl #self_type {
            #[uniffi::constructor]
            pub fn #method_ident(handler: Box<dyn #handler_ident>) -> std::sync::Arc<Self> {
                let handler: std::sync::Arc<dyn #handler_ident> = std::sync::Arc::from(handler);
                let callback: Box<dyn Fn( #( #callback_inputs ),* ) + Send + Sync> =
                    Box::new(move | #( #arg_idents: #callback_inputs ),* | {
                        handler.on_event( #( #arg_idents ),* );
                    });
                std::sync::Arc::new(#body)
            }
        }
    }
}

fn self_type_ident(self_type: &syn::Type) -> Option<Ident> {
    match self_type {
        syn::Type::Path(path) => path.path.segments.last().map(|segment| segment.ident.clone()),
        _ => None,
    }
}
