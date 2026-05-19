use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Data, DataStruct, DeriveInput, Field, Fields, FieldsNamed, Generics, ItemStruct, Path, Token, Type,
    parse_macro_input, parse_quote, punctuated::Punctuated, spanned::Spanned,
};

// TODO: This is very hacky, there has to be a better way to do all of this stuff

pub fn uzu_config_abstract(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    let concrete_configs = parse_macro_input!(args with Punctuated::<Path, Token![,]>::parse_terminated);
    let input = parse_macro_input!(input as ItemStruct);

    let struct_fields = match input.fields {
        Fields::Named(FieldsNamed {
            named,
            ..
        }) => named,
        Fields::Unit => Punctuated::new(),
        fields => {
            return syn::Error::new(fields.span(), "uzu_config_abstract requires a named-field or unit struct")
                .to_compile_error()
                .into();
        },
    };

    let input_vis = input.vis;
    let struct_ident = input.ident;
    let enum_ident = format_ident!("Any{}", struct_ident);
    let (alias_struct_fields, callback_struct_fields, enum_field_getters) = struct_fields
        .into_iter()
        .map(|mut struct_field| {
            let field_ident = struct_field.ident.as_ref().unwrap();
            let typealias_ident = format_ident!("__{}_{}", struct_ident, field_ident);
            let field_type = std::mem::replace(
                &mut struct_field.ty,
                Type::Verbatim(quote! { $($typealias_base)::*::#typealias_ident }),
            );
            let match_arms = concrete_configs.iter().map(|concrete_config_path| {
                let concrete_config_ident = &concrete_config_path.segments.last().unwrap().ident;
                quote! { Self::#concrete_config_ident(__concrete) => &__concrete.#field_ident, }
            });
            (
                quote! {
                    #[allow(non_camel_case_types)]
                    type #typealias_ident = #field_type;
                },
                quote! {
                    #struct_field
                },
                quote! {
                    #input_vis fn #field_ident(&self) -> &#field_type {
                        match self {
                            #(#match_arms)*
                        }
                    }
                },
            )
        })
        .collect::<(Vec<_>, Vec<_>, Vec<_>)>();
    let enum_variants = concrete_configs.into_iter().map(|concrete_config_path| {
        let concrete_config_ident = &concrete_config_path.segments.last().unwrap().ident;
        quote! { #concrete_config_ident(#concrete_config_path) }
    });

    quote! {
        #(#alias_struct_fields)*

        macro_rules! #struct_ident {
            ($($typealias_base:ident)::+, $callback:path) => {
                $callback! {
                    #(#callback_struct_fields,)*
                }
            };
        }

        use #struct_ident;

        #[proc_macros::uzu_config]
        #[serde(untagged)]
        #input_vis enum #enum_ident {
            #(#enum_variants,)*
        }

        impl #enum_ident {
            #(#enum_field_getters)*
        }
    }
    .into()
}

pub fn uzu_config(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    let mut input = parse_macro_input!(input as DeriveInput);

    if !args.is_empty() {
        let fields_macro_path = parse_macro_input!(args as Path);
        let fields_macro_base_path = {
            let mut fields_macro_base = fields_macro_path.clone();
            fields_macro_base.segments.pop();
            fields_macro_base.segments.pop_punct();
            fields_macro_base
        };
        let fields_macro_base = if fields_macro_base_path.segments.is_empty() {
            quote! { self }
        } else {
            quote! { #fields_macro_base_path }
        };

        let struct_fields = match input.data {
            Data::Struct(DataStruct {
                fields: Fields::Named(FieldsNamed {
                    named,
                    ..
                }),
                ..
            }) => named,
            Data::Struct(DataStruct {
                fields: Fields::Unit,
                ..
            }) => Punctuated::new(),
            _ => {
                return syn::Error::new(
                    input.ident.span(),
                    "uzu_config(Abstract) requires a named-field or unit struct",
                )
                .to_compile_error()
                .into();
            },
        };

        let input_attrs = input.attrs;
        let input_vis = input.vis;
        let input_ident = input.ident;
        let input_ident_str = input_ident.to_string();
        let input_generics = input.generics;
        let struct_fields = struct_fields.into_iter().collect::<Vec<_>>();
        let callback_ident = format_ident!("__emit_{}", input_ident);

        // Re-entry with super fields inserted
        return quote! {
            macro_rules! #callback_ident {
                ($($super_fields:tt)*) => {
                    #[proc_macros::uzu_config]
                    #(#input_attrs)*
                    #input_vis struct #input_ident #input_generics {
                        #[serde(rename = "type")]
                        ty: monostate::MustBe!(#input_ident_str),
                        $($super_fields)*
                        #(#struct_fields,)*
                    }
                }
            }

            #fields_macro_path!(#fields_macro_base, #callback_ident);
        }
        .into();
    };

    let fields: Vec<&mut Field> = match &mut input.data {
        Data::Struct(data_struct) => data_struct.fields.iter_mut().collect(),
        Data::Enum(data_enum) => data_enum.variants.iter_mut().flat_map(|variant| variant.fields.iter_mut()).collect(),
        Data::Union(_) => panic!("Unions are not supported by uzu_config!"),
    };

    for field in fields {
        field.attrs.push(parse_quote!(#[serde(deserialize_with = "crate::utils::strict_serde::required")]));
    }

    let input_ident = &input.ident;
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();

    let impl_generics = impl_generics.into_token_stream().into();
    let mut impl_generics = parse_macro_input!(impl_generics as Generics);
    impl_generics.params.insert(0, parse_quote!('__de));

    quote! {
        #[derive(Debug, Clone, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        #input

        impl #impl_generics crate::utils::strict_serde::DeserializeStrict<'__de> for #input_ident #type_generics #where_clause {}
    }
    .into()
}
