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
        BindingKind::Class => quote! {
            #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Object))]
        },
        BindingKind::ClassCloneable => quote! {
            #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Record))]
        },
        BindingKind::Implementation => quote! {
            #[cfg_attr(feature = "bindings-uniffi", uniffi::export)]
        },
        _ => quote! {},
    }
}

pub fn method_attribute(flavor: &MethodFlavor) -> Option<Attribute> {
    match flavor {
        MethodFlavor::Plain | MethodFlavor::Getter | MethodFlavor::Setter => None,
        MethodFlavor::Constructor | MethodFlavor::Factory => Some(parse_quote! {
            #[cfg_attr(feature = "bindings-uniffi", uniffi::constructor)]
        }),
    }
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

    quote! {
        #[cfg(feature = "bindings-uniffi")]
        #[uniffi::export(callback_interface)]
        pub trait #handler_ident: Send + Sync {
            fn on_event(&self);
        }

        #[cfg(feature = "bindings-uniffi")]
        #[uniffi::export]
        impl #self_type {
            #[uniffi::constructor]
            pub fn #method_ident(handler: Box<dyn #handler_ident>) -> std::sync::Arc<Self> {
                let handler: std::sync::Arc<dyn #handler_ident> = std::sync::Arc::from(handler);
                let callback: Box<dyn Fn() + Send + Sync> = Box::new(move || handler.on_event());
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
