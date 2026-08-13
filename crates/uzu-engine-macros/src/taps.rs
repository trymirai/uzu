use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Attribute, Ident, LitStr, Token, Visibility, braced,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
};

enum FieldKind {
    Array,
    Sub(Ident),
    Repeated(Ident),
}

struct TapField {
    name: Ident,
    kind: FieldKind,
    rename: Option<LitStr>,
    skip: bool,
    flatten: bool,
}

struct TapStruct {
    visibility: Visibility,
    name: Ident,
    fields: Punctuated<TapField, Token![,]>,
}

impl Parse for TapField {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let TapAttrs {
            rename,
            skip,
            flatten,
        } = parse_tap_attrs(&attrs)?;
        let name = input.parse()?;

        let kind = if input.peek(Token![:]) {
            input.parse::<Token![:]>()?;
            if input.peek(syn::token::Bracket) {
                let content;
                syn::bracketed!(content in input);
                FieldKind::Repeated(content.parse()?)
            } else {
                FieldKind::Sub(input.parse()?)
            }
        } else {
            FieldKind::Array
        };

        Ok(Self {
            name,
            kind,
            rename,
            skip,
            flatten,
        })
    }
}

impl Parse for TapStruct {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let visibility = input.parse()?;
        let name = input.parse()?;
        let content;
        braced!(content in input);

        Ok(Self {
            visibility,
            name,
            fields: content.parse_terminated(TapField::parse, Token![,])?,
        })
    }
}

#[derive(Default)]
struct TapAttrs {
    rename: Option<LitStr>,
    skip: bool,
    flatten: bool,
}

fn parse_tap_attrs(attrs: &[Attribute]) -> syn::Result<TapAttrs> {
    let mut parsed = TapAttrs::default();
    for attr in attrs.iter().filter(|attr| attr.path().is_ident("tap")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                parsed.skip = true;
                return Ok(());
            }
            if meta.path.is_ident("flatten") {
                parsed.flatten = true;
                return Ok(());
            }
            if meta.path.is_ident("rename") {
                parsed.rename = Some(meta.value()?.parse()?);
                return Ok(());
            }
            Err(meta.error("expected `skip`, `flatten` or `rename = \"...\"`"))
        })?;
    }
    Ok(parsed)
}

pub fn taps(input: TokenStream) -> TokenStream {
    let TapStruct {
        visibility,
        name,
        fields,
    } = parse_macro_input!(input as TapStruct);
    let request_name = format_ident!("{}Request", name);

    let mut tap_fields = Vec::new();
    let mut tap_defaults = Vec::new();
    let mut request_fields = Vec::new();
    let mut none_fields = Vec::new();
    let mut all_fields = Vec::new();
    let mut collect_arms = Vec::new();

    for field in &fields {
        let field_name = &field.name;
        let segment = field.rename.as_ref().map(LitStr::value).unwrap_or_else(|| field_name.to_string());

        match &field.kind {
            FieldKind::Array => {
                tap_fields.push(quote! { pub #field_name: Option<crate::trace::Array<B>> });
                tap_defaults.push(quote! { #field_name: None });
                request_fields.push(quote! { pub #field_name: bool });
                none_fields.push(quote! { #field_name: false });
                all_fields.push(quote! { #field_name: true });
            },
            FieldKind::Sub(sub) | FieldKind::Repeated(sub) => {
                let sub_request = format_ident!("{}Request", sub);
                let tap_type = match &field.kind {
                    FieldKind::Repeated(_) => quote! { Vec<#sub<B>> },
                    _ => quote! { Option<#sub<B>> },
                };
                tap_defaults.push(match &field.kind {
                    FieldKind::Repeated(_) => quote! { #field_name: Vec::new() },
                    _ => quote! { #field_name: None },
                });
                tap_fields.push(quote! { pub #field_name: #tap_type });
                request_fields.push(quote! { pub #field_name: Option<#sub_request> });
                none_fields.push(quote! { #field_name: None });
                all_fields.push(quote! { #field_name: Some(#sub_request::all()) });
            },
        }

        if field.skip {
            continue;
        }

        collect_arms.push(match &field.kind {
            FieldKind::Array => quote! {
                if let Some(array) = &self.#field_name {
                    out.push((format!("{prefix}{}", #segment), array));
                }
            },
            // `flatten` contributes no segment, so a sub-tap's fields land directly
            // in the parent's namespace.
            FieldKind::Sub(_) if field.flatten => quote! {
                if let Some(sub) = &self.#field_name {
                    sub.collect(prefix, out);
                }
            },
            FieldKind::Sub(_) => quote! {
                if let Some(sub) = &self.#field_name {
                    sub.collect(&format!("{prefix}{}.", #segment), out);
                }
            },
            FieldKind::Repeated(_) => quote! {
                for (index, sub) in self.#field_name.iter().enumerate() {
                    sub.collect(&format!("{prefix}{}.{index}.", #segment), out);
                }
            },
        });
    }

    quote! {
        #visibility struct #name<B: crate::backends::common::Backend> {
            #(#tap_fields,)*
        }

        // Hand-written so the backend parameter does not pick up a `Default` bound.
        impl<B: crate::backends::common::Backend> Default for #name<B> {
            fn default() -> Self {
                Self {
                    #(#tap_defaults,)*
                }
            }
        }

        impl<B: crate::backends::common::Backend> #name<B> {
            /// Appends every captured array as `(path, array)`. Field names are the
            /// path segments, so the struct tree is the safetensors layout.
            pub fn collect<'a>(
                &'a self,
                prefix: &str,
                out: &mut Vec<(String, &'a crate::trace::Array<B>)>,
            ) {
                #(#collect_arms)*
            }

            pub fn write(
                &self,
                output_path: &std::path::Path,
                metadata: Option<std::collections::HashMap<String, String>>,
            ) -> Result<(), crate::trace::Error> {
                let mut arrays = Vec::new();
                self.collect("", &mut arrays);
                ::safetensors::serialize_to_file(
                    arrays.iter().map(|(path, array)| (path.as_str(), *array)),
                    metadata,
                    output_path,
                )?;

                Ok(())
            }

            pub fn len(&self) -> usize {
                let mut arrays = Vec::new();
                self.collect("", &mut arrays);
                arrays.len()
            }

            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }
        }

        #[derive(Debug, Default, Clone, PartialEq, Eq)]
        #visibility struct #request_name {
            #(#request_fields,)*
        }

        impl #request_name {
            /// Nothing captured. Encode paths fall back to this when given no request.
            pub const NONE: Self = Self {
                #(#none_fields,)*
            };

            /// Every array in this subtree, recursively.
            pub fn all() -> Self {
                Self {
                    #(#all_fields,)*
                }
            }
        }
    }
    .into()
}
