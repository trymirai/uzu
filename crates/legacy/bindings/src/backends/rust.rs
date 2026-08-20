use proc_macro2::TokenStream;
use quote::quote;

use crate::{
    backends::Backend,
    contexts::{ImplementationContext, MethodMetadata},
    types::MethodFlavor,
};

pub struct Rust;

impl Backend for Rust {
    fn method_companions(
        context: &ImplementationContext,
        metadata: &MethodMetadata,
    ) -> TokenStream {
        match metadata.flavor {
            MethodFlavor::Factory => {
                let self_type = &context.self_type;
                let method = &metadata.method;
                quote! {
                    impl #self_type {
                        #method
                    }
                }
            },
            _ => quote! {},
        }
    }
}
