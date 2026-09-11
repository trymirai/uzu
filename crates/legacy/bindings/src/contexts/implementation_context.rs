use proc_macro2::TokenStream;
use quote::quote;
use syn::{Attribute, ImplItem, ItemImpl, parse::Parser};

use crate::{
    backends::{Backend, Napi, Pyo3, Uniffi, Wasm},
    contexts::MethodMetadata,
    types::{Kind, MethodFlavor},
};

pub struct ImplementationContext {
    pub self_type: syn::Type,
    pub item: ItemImpl,
    pub methods: Vec<MethodMetadata>,
}

impl ImplementationContext {
    pub fn from_token_stream(tokens: TokenStream) -> syn::Result<Self> {
        let mut item: ItemImpl = syn::parse2(tokens)?;
        let self_type = (*item.self_ty).clone();

        let mut methods: Vec<MethodMetadata> = Vec::new();
        let mut retained_items: Vec<ImplItem> = Vec::new();

        for impl_item in std::mem::take(&mut item.items) {
            match impl_item {
                ImplItem::Fn(mut method) => {
                    let flavor = parse_method_flavor(&method).ok_or_else(|| {
                        syn::Error::new_spanned(
                            &method.sig,
                            format!(
                                "Method `{}` inside `bindings::export(Implementation)` must be annotated with `#[bindings::export(Method(...))]`",
                                method.sig.ident
                            ),
                        )
                    })?;
                    strip_export_attributes(&mut method.attrs);

                    let extracted = matches!(flavor, MethodFlavor::Factory | MethodFlavor::FactoryWithCallback);

                    if !extracted {
                        let metadata_for_attrs = MethodMetadata {
                            method: method.clone(),
                            flavor,
                        };
                        apply_backend_method_attributes(&mut method, &metadata_for_attrs)?;
                    }

                    methods.push(MethodMetadata {
                        method: method.clone(),
                        flavor,
                    });

                    if !extracted {
                        retained_items.push(ImplItem::Fn(method));
                    }
                },
                other => retained_items.push(other),
            }
        }
        item.items = retained_items;

        Ok(Self {
            self_type,
            item,
            methods,
        })
    }

    pub fn all_methods(&self) -> impl Iterator<Item = &MethodMetadata> {
        self.methods.iter()
    }

    pub fn methods_of(
        &self,
        flavor: MethodFlavor,
    ) -> impl Iterator<Item = &MethodMetadata> {
        self.methods.iter().filter(move |metadata| metadata.flavor == flavor)
    }

    pub fn self_type_ident(&self) -> Option<&syn::Ident> {
        match &self.self_type {
            syn::Type::Path(type_path) => type_path.path.segments.last().map(|segment| &segment.ident),
            _ => None,
        }
    }

    pub fn has_async_method(&self) -> bool {
        self.item
            .items
            .iter()
            .any(|impl_item| matches!(impl_item, syn::ImplItem::Fn(method) if method.sig.asyncness.is_some()))
    }
}

fn apply_backend_method_attributes(
    method: &mut syn::ImplItemFn,
    metadata: &MethodMetadata,
) -> syn::Result<()> {
    let napi_tokens = Napi::method_attributes(metadata);
    let pyo3_tokens = Pyo3::method_attributes(metadata);
    let uniffi_tokens = Uniffi::method_attributes(metadata);
    let wasm_tokens = Wasm::method_attributes(metadata);
    let combined = quote! {
        #napi_tokens
        #pyo3_tokens
        #uniffi_tokens
        #wasm_tokens
    };
    let parsed_attributes = Attribute::parse_outer.parse2(combined)?;
    method.attrs.extend(parsed_attributes);
    Ok(())
}

fn parse_method_flavor(method: &syn::ImplItemFn) -> Option<MethodFlavor> {
    method.attrs.iter().find_map(|attribute| {
        if !is_export_attribute(attribute) {
            return None;
        }
        let kind: Kind = attribute.parse_args().ok()?;
        match kind {
            Kind::Method(flavor) => Some(flavor),
            _ => None,
        }
    })
}

fn strip_export_attributes(attributes: &mut Vec<Attribute>) {
    attributes.retain(|attribute| !is_export_attribute(attribute));
}

fn is_export_attribute(attribute: &Attribute) -> bool {
    let path = attribute.path();
    let Some(last) = path.segments.last() else {
        return false;
    };
    if last.ident != "export" {
        return false;
    }
    matches!(
        path.segments.first(),
        Some(first) if first.ident == "bindings" || first.ident == "export"
    )
}
