use quote::{format_ident, quote};
use syn::{Attribute, Ident, Token, Variant, parse_quote, punctuated::Punctuated};

use crate::{BindingKind, MethodFlavor};

pub fn attributes(kind: &BindingKind) -> proc_macro2::TokenStream {
    match kind {
        BindingKind::Enum => quote! {
            #[cfg_attr(feature = "bindings-napi", napi_derive::napi(string_enum))]
        },
        BindingKind::Class => quote! {
            #[cfg_attr(feature = "bindings-napi", napi_derive::napi)]
        },
        BindingKind::ClassCloneable => quote! {
            #[cfg_attr(feature = "bindings-napi", napi_derive::napi(constructor))]
        },
        BindingKind::Stream => quote! {
            #[cfg_attr(feature = "bindings-napi", napi_derive::napi(async_iterator))]
        },
        BindingKind::Struct => quote! {
            #[cfg_attr(feature = "bindings-napi", napi_derive::napi(object))]
        },
        BindingKind::Implementation => quote! {
            #[cfg_attr(feature = "bindings-napi", napi_derive::napi)]
        },
        BindingKind::Alias => quote! {
            #[cfg_attr(feature = "bindings-napi", napi_derive::napi)]
        },
        _ => quote! {},
    }
}

pub fn method_attribute(flavor: &MethodFlavor) -> Attribute {
    match flavor {
        MethodFlavor::Plain => parse_quote! {
            #[cfg_attr(feature = "bindings-napi", napi)]
        },
        MethodFlavor::Constructor => parse_quote! {
            #[cfg_attr(feature = "bindings-napi", napi(constructor))]
        },
        MethodFlavor::Factory => parse_quote! {
            #[cfg_attr(feature = "bindings-napi", napi)]
        },
        MethodFlavor::Getter => parse_quote! {
            #[cfg_attr(feature = "bindings-napi", napi(getter))]
        },
        MethodFlavor::Setter => parse_quote! {
            #[cfg_attr(feature = "bindings-napi", napi(setter))]
        },
    }
}

pub fn struct_value_implementations(type_name: &Ident) -> proc_macro2::TokenStream {
    quote! {
        #[cfg(feature = "bindings-napi")]
        const _: () = {
            use napi::bindgen_prelude::ToNapiValue;
            use napi::sys::{napi_env, napi_value};

            impl ToNapiValue for &#type_name {
                unsafe fn to_napi_value(env: napi_env, value: Self) -> napi::Result<napi_value> {
                    let owned: #type_name = ::core::clone::Clone::clone(value);
                    <#type_name as ToNapiValue>::to_napi_value(env, owned)
                }
            }

            impl ToNapiValue for &mut #type_name {
                unsafe fn to_napi_value(env: napi_env, value: Self) -> napi::Result<napi_value> {
                    let owned: #type_name = ::core::clone::Clone::clone(value);
                    <#type_name as ToNapiValue>::to_napi_value(env, owned)
                }
            }
        };
    }
}

pub fn class_value_implementations(type_name: &Ident) -> proc_macro2::TokenStream {
    quote! {
        #[cfg(feature = "bindings-napi")]
        const _: () = {
            use napi::bindgen_prelude::{FromNapiValue, ToNapiValue};
            use napi::sys::{napi_env, napi_value};

            impl ToNapiValue for &#type_name {
                unsafe fn to_napi_value(env: napi_env, value: Self) -> napi::Result<napi_value> {
                    let owned: #type_name = ::core::clone::Clone::clone(value);
                    <#type_name as ToNapiValue>::to_napi_value(env, owned)
                }
            }

            impl ToNapiValue for &mut #type_name {
                unsafe fn to_napi_value(env: napi_env, value: Self) -> napi::Result<napi_value> {
                    let owned: #type_name = ::core::clone::Clone::clone(value);
                    <#type_name as ToNapiValue>::to_napi_value(env, owned)
                }
            }

            impl FromNapiValue for #type_name {
                unsafe fn from_napi_value(env: napi_env, value: napi_value) -> napi::Result<Self> {
                    let reference = <&#type_name as FromNapiValue>::from_napi_value(env, value)?;
                    Ok(::core::clone::Clone::clone(reference))
                }
            }
        };
    }
}

pub fn enum_variant_classes(
    enum_ident: &Ident,
    variants: &Punctuated<Variant, Token![,]>,
) -> proc_macro2::TokenStream {
    let union_alias_ident = format_ident!("{}Napi", enum_ident);
    let variant_count = variants.len();
    let either_type_ident: Option<Ident> = match variant_count {
        1 => None,
        2 => Some(format_ident!("Either")),
        n if (3..=26).contains(&n) => Some(format_ident!("Either{n}")),
        n => {
            return syn::Error::new_spanned(
                enum_ident,
                format!("bindings::export(Enum) supports up to 26 data variants (found {n})"),
            )
            .to_compile_error();
        },
    };
    let either_variant_labels: Vec<Ident> =
        (0..variant_count).map(|index| format_ident!("{}", (b'A' + index as u8) as char)).collect();

    let mut variant_class_items = Vec::new();
    let mut from_variant_impls = Vec::new();
    let mut to_napi_match_arms = Vec::new();
    let mut from_napi_try_branches = Vec::new();
    let mut union_type_params = Vec::new();
    let mut enum_to_union_arms = Vec::new();
    let mut union_to_enum_arms = Vec::new();

    for (variant_index, variant) in variants.iter().enumerate() {
        let variant_ident = &variant.ident;
        let variant_class_ident = format_ident!("{}{}", enum_ident, variant_ident);
        let either_label = &either_variant_labels[variant_index];
        union_type_params.push(quote! { #variant_class_ident });
        let named_fields = match &variant.fields {
            syn::Fields::Named(named) => &named.named,
            _ => unreachable!("non-named variants must be rejected before reaching this function"),
        };
        let field_declarations: Vec<proc_macro2::TokenStream> = named_fields
            .iter()
            .map(|field| {
                let field_ident = field.ident.as_ref().expect("named field");
                let field_type = &field.ty;
                quote! { pub #field_ident: #field_type }
            })
            .collect();
        let field_idents: Vec<&Ident> =
            named_fields.iter().map(|field| field.ident.as_ref().expect("named field")).collect();

        let variant_class_value_impls = class_value_implementations(&variant_class_ident);
        variant_class_items.push(quote! {
            #[cfg(feature = "bindings-napi")]
            #[napi_derive::napi(constructor)]
            #[derive(Clone)]
            pub struct #variant_class_ident {
                #( #field_declarations ),*
            }

            #variant_class_value_impls
        });

        from_variant_impls.push(quote! {
            #[cfg(feature = "bindings-napi")]
            impl From<#variant_class_ident> for #enum_ident {
                fn from(value: #variant_class_ident) -> Self {
                    #enum_ident::#variant_ident {
                        #( #field_idents: value.#field_idents ),*
                    }
                }
            }
        });

        to_napi_match_arms.push(quote! {
            #enum_ident::#variant_ident { #( #field_idents ),* } => {
                ToNapiValue::to_napi_value(
                    env,
                    #variant_class_ident { #( #field_idents ),* },
                )
            }
        });

        from_napi_try_branches.push(quote! {
            if <&#variant_class_ident as napi::bindgen_prelude::ValidateNapiValue>::validate(env, val).is_ok() {
                let instance = ClassInstance::<#variant_class_ident>::from_napi_value(env, val)?;
                let inner: &#variant_class_ident =
                    <ClassInstance<#variant_class_ident> as ::core::ops::Deref>::deref(&instance);
                return Ok(#enum_ident::#variant_ident {
                    #( #field_idents: ::core::clone::Clone::clone(&inner.#field_idents) ),*
                });
            }
        });

        let wrap_expression = match &either_type_ident {
            Some(either_ident) => quote! {
                napi::bindgen_prelude::#either_ident::#either_label(
                    #variant_class_ident { #( #field_idents ),* },
                )
            },
            None => quote! {
                #variant_class_ident { #( #field_idents ),* }
            },
        };
        enum_to_union_arms.push(quote! {
            #enum_ident::#variant_ident { #( #field_idents ),* } => { #wrap_expression }
        });

        let unwrap_pattern = match &either_type_ident {
            Some(either_ident) => quote! {
                napi::bindgen_prelude::#either_ident::#either_label(inner)
            },
            None => quote! { inner },
        };
        union_to_enum_arms.push(quote! {
            #unwrap_pattern => { #enum_ident::from(inner) }
        });
    }

    let expected_message = format!("Expected instance of variant class for {}", enum_ident);

    let union_js_name = enum_ident.to_string();
    let union_alias_definition = match &either_type_ident {
        Some(either_ident) => quote! {
            #[cfg(feature = "bindings-napi")]
            #[napi_derive::napi(js_name = #union_js_name)]
            pub type #union_alias_ident =
                napi::bindgen_prelude::#either_ident<#( #union_type_params ),*>;
        },
        None => {
            let single_variant_class = &union_type_params[0];
            quote! {
                #[cfg(feature = "bindings-napi")]
                #[napi_derive::napi(js_name = #union_js_name)]
                pub type #union_alias_ident = #single_variant_class;
            }
        },
    };

    let union_from_impls = if either_type_ident.is_some() {
        quote! {
            #[cfg(feature = "bindings-napi")]
            impl From<#enum_ident> for #union_alias_ident {
                fn from(value: #enum_ident) -> Self {
                    match value {
                        #( #enum_to_union_arms ),*
                    }
                }
            }

            #[cfg(feature = "bindings-napi")]
            impl From<#union_alias_ident> for #enum_ident {
                fn from(value: #union_alias_ident) -> Self {
                    match value {
                        #( #union_to_enum_arms ),*
                    }
                }
            }
        }
    } else {
        quote! {}
    };

    quote! {
        #( #variant_class_items )*
        #( #from_variant_impls )*

        #union_alias_definition
        #union_from_impls

        #[cfg(feature = "bindings-napi")]
        const _: () = {
            use napi::bindgen_prelude::{ClassInstance, FromNapiValue, ToNapiValue};
            use napi::sys::{napi_env, napi_value};

            impl FromNapiValue for #enum_ident {
                unsafe fn from_napi_value(
                    env: napi_env,
                    val: napi_value,
                ) -> napi::Result<Self> {
                    #( #from_napi_try_branches )*
                    Err(napi::Error::from_reason(#expected_message))
                }
            }

            impl ToNapiValue for #enum_ident {
                unsafe fn to_napi_value(
                    env: napi_env,
                    val: Self,
                ) -> napi::Result<napi_value> {
                    match val {
                        #( #to_napi_match_arms ),*
                    }
                }
            }

            impl ToNapiValue for &#enum_ident {
                unsafe fn to_napi_value(
                    env: napi_env,
                    val: Self,
                ) -> napi::Result<napi_value> {
                    let owned: #enum_ident = ::core::clone::Clone::clone(val);
                    <#enum_ident as ToNapiValue>::to_napi_value(env, owned)
                }
            }

            impl ToNapiValue for &mut #enum_ident {
                unsafe fn to_napi_value(
                    env: napi_env,
                    val: Self,
                ) -> napi::Result<napi_value> {
                    let owned: #enum_ident = ::core::clone::Clone::clone(val);
                    <#enum_ident as ToNapiValue>::to_napi_value(env, owned)
                }
            }
        };
    }
}

pub fn stream_next_generator(
    self_type: &syn::Type,
    method: &syn::ImplItemFn,
) -> proc_macro2::TokenStream {
    let method_ident = &method.sig.ident;
    let yield_type = match extract_option_inner_type(&method.sig.output) {
        Some(ty) => ty,
        None => {
            return syn::Error::new_spanned(
                &method.sig,
                "bindings::export(StreamNext) requires return type `Option<T>`",
            )
            .to_compile_error();
        },
    };

    quote! {
        #[cfg(feature = "bindings-napi")]
        #[napi_derive::napi]
        impl napi::bindgen_prelude::AsyncGenerator for #self_type {
            type Yield = #yield_type;
            type Next = ();
            type Return = ();

            fn next(
                &mut self,
                _value: Option<Self::Next>,
            ) -> impl ::core::future::Future<Output = napi::Result<Option<Self::Yield>>>
                + Send
                + 'static {
                let this: Self = ::core::clone::Clone::clone(self);
                async move { Ok(this.#method_ident().await) }
            }
        }
    }
}

fn extract_option_inner_type(output: &syn::ReturnType) -> Option<syn::Type> {
    let return_type = match output {
        syn::ReturnType::Type(_, return_type) => return_type.as_ref(),
        syn::ReturnType::Default => return None,
    };
    let type_path = match return_type {
        syn::Type::Path(type_path) => type_path,
        _ => return None,
    };
    let last_segment = type_path.path.segments.last()?;
    if last_segment.ident != "Option" {
        return None;
    }
    let generic_arguments = match &last_segment.arguments {
        syn::PathArguments::AngleBracketed(angle_bracketed) => &angle_bracketed.args,
        _ => return None,
    };
    for argument in generic_arguments {
        if let syn::GenericArgument::Type(inner_type) = argument {
            return Some(inner_type.clone());
        }
    }
    None
}

pub fn factory_with_callback(
    self_type: &syn::Type,
    method: &syn::ImplItemFn,
) -> proc_macro2::TokenStream {
    let method_ident = &method.sig.ident;
    let body = &method.block;
    quote! {
        #[cfg(feature = "bindings-napi")]
        #[napi_derive::napi]
        impl #self_type {
            #[napi(factory)]
            pub fn #method_ident(
                callback: napi::bindgen_prelude::Function<'static, (), ()>,
            ) -> napi::Result<Self> {
                let threadsafe_function = callback
                    .build_threadsafe_function()
                    .callee_handled::<false>()
                    .weak::<true>()
                    .build()
                    .map_err(|error| napi::Error::from_reason(error.to_string()))?;
                let callback: Box<dyn Fn() + Send + Sync> = Box::new(move || {
                    let _ = threadsafe_function.call(
                        (),
                        napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                    );
                });
                Ok(#body)
            }
        }
    }
}

pub fn error_implementations(type_name: &Ident) -> proc_macro2::TokenStream {
    quote! {
        #[cfg(feature = "bindings-napi")]
        impl From<#type_name> for napi::Error {
            fn from(error: #type_name) -> Self {
                napi::Error::from_reason(error.to_string())
            }
        }

        #[cfg(feature = "bindings-napi")]
        impl From<#type_name> for napi::JsError {
            fn from(error: #type_name) -> Self {
                let napi_error: napi::Error = error.into();
                napi::JsError::from(napi_error)
            }
        }
    }
}
