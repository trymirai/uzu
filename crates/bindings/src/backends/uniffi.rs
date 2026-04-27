use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

use crate::{
    backends::Backend,
    contexts::{
        ClassContext, EnumerationContext, ErrorContext, ImplementationContext, MethodMetadata, StructureContext,
    },
    types::MethodFlavor,
};

pub struct Uniffi;

impl Backend for Uniffi {
    fn structure_attributes(_context: &StructureContext) -> TokenStream {
        quote! {
            #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Record))]
        }
    }

    fn class_attributes(_context: &ClassContext) -> TokenStream {
        quote! {
            #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Object))]
        }
    }

    fn enumeration_attributes(_context: &EnumerationContext) -> TokenStream {
        quote! {
            #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Enum))]
        }
    }

    fn implementation_attributes(context: &ImplementationContext) -> TokenStream {
        if context.has_async_method() {
            quote! {
                #[cfg_attr(feature = "bindings-uniffi", uniffi::export(async_runtime = "tokio"))]
            }
        } else {
            quote! {
                #[cfg_attr(feature = "bindings-uniffi", uniffi::export)]
            }
        }
    }

    fn method_attributes(metadata: &MethodMetadata) -> TokenStream {
        match metadata.flavor {
            MethodFlavor::Constructor => quote! {
                #[cfg_attr(feature = "bindings-uniffi", uniffi::constructor)]
            },
            _ => quote! {},
        }
    }

    fn method_companions(
        context: &ImplementationContext,
        metadata: &MethodMetadata,
    ) -> TokenStream {
        match metadata.flavor {
            MethodFlavor::Factory => factory_expansion(context, metadata),
            MethodFlavor::FactoryWithCallback => factory_with_callback_expansion(context, metadata),
            _ => quote! {},
        }
    }

    fn error_attributes(_context: &ErrorContext) -> TokenStream {
        quote! {
            #[cfg_attr(feature = "bindings-uniffi", derive(uniffi::Error))]
        }
    }
}

fn factory_expansion(
    context: &ImplementationContext,
    metadata: &MethodMetadata,
) -> TokenStream {
    let self_type = &context.self_type;
    let method = &metadata.method;
    let method_ident = &method.sig.ident;
    let self_ident = match context.self_type_ident() {
        Some(ident) => ident,
        None => {
            return syn::Error::new_spanned(self_type, "Bindings::export(Method(Factory)) requires a named self type")
                .to_compile_error();
        },
    };
    let fn_name = format_ident!("{}_{}", heck::AsSnakeCase(self_ident.to_string()).to_string(), method_ident);
    let inputs = &method.sig.inputs;
    let output = replace_self_in_return(&method.sig.output, self_type);
    let asyncness = &method.sig.asyncness;
    let arg_idents = metadata.arg_idents();
    let forward_await = asyncness.as_ref().map(|_| quote! { .await });
    let export_attribute = if asyncness.is_some() {
        quote! { #[uniffi::export(async_runtime = "tokio")] }
    } else {
        quote! { #[uniffi::export] }
    };

    quote! {
        #[cfg(feature = "bindings-uniffi")]
        #export_attribute
        pub #asyncness fn #fn_name( #inputs ) #output {
            <#self_type>::#method_ident( #( #arg_idents ),* )#forward_await
        }
    }
}

fn factory_with_callback_expansion(
    context: &ImplementationContext,
    metadata: &MethodMetadata,
) -> TokenStream {
    let self_type = &context.self_type;
    let method = &metadata.method;
    let method_ident = &method.sig.ident;
    let body = &method.block;

    let self_ident = match context.self_type_ident() {
        Some(ident) => ident,
        None => {
            return syn::Error::new_spanned(
                self_type,
                "Bindings::export(Method(FactoryWithCallback)) requires a named self type",
            )
            .to_compile_error();
        },
    };
    let handler_ident = format_ident!("{}Handler", self_ident);
    let callback_inputs = match metadata.callback_inputs() {
        Some(inputs) => inputs,
        None => {
            return syn::Error::new_spanned(
                &method.sig,
                "Bindings::export(Method(FactoryWithCallback)) requires a parameter of type \
                 `Box<dyn Fn(..) + Send + Sync>` (return type must be `()`)",
            )
            .to_compile_error();
        },
    };
    let arg_idents: Vec<Ident> = (0..callback_inputs.len()).map(|index| format_ident!("arg{index}")).collect();

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
