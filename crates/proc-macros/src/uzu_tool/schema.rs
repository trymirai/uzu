use proc_macro::TokenStream;
use quote::quote;
use serde_derive_internals::{
    Ctxt, Derive,
    ast::{Container as SerdeContainer, Data as SerdeData},
};
use syn::{__private::TokenStream2, Data, DeriveInput, Error, Fields, parse_macro_input};

use crate::uzu_tool::util::{doc_string, is_option, nagare_path};

pub fn uzu_derive_tool_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_tool_schema(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn expand_tool_schema(input: DeriveInput) -> syn::Result<TokenStream2> {
    let nagare = nagare_path()?;

    let Data::Struct(data) = &input.data else {
        return Err(Error::new_spanned(&input.ident, "#[derive(UzuToolSchema)] only supports structs"));
    };
    let Fields::Named(_) = &data.fields else {
        return Err(Error::new_spanned(
            &input.ident,
            "#[derive(UzuToolSchema)] only supports structs with named fields",
        ));
    };

    let serde_context = Ctxt::new();
    let serde_container = SerdeContainer::from_ast(&serde_context, &input, Derive::Deserialize);
    serde_context.check()?;
    let Some(serde_container) = serde_container else {
        return Err(Error::new_spanned(&input.ident, "failed to parse Serde attributes"));
    };
    let SerdeData::Struct(_, serde_fields) = &serde_container.data else {
        unreachable!();
    };

    if serde_container.attrs.transparent()
        || serde_container.attrs.type_from().is_some()
        || serde_container.attrs.type_try_from().is_some()
        || serde_container.attrs.type_into().is_some()
        || serde_container.attrs.remote().is_some()
    {
        return Err(Error::new_spanned(
            &input.ident,
            "UzuToolSchema does not support Serde container attributes that change the serialized representation",
        ));
    }

    let schema_fields = serde_fields
        .iter()
        .filter_map(|field| {
            let attrs = &field.attrs;
            if attrs.skip_serializing() != attrs.skip_deserializing() {
                return Some(Err(Error::new_spanned(
                    field.original,
                    "UzuToolSchema requires fields to be skipped in both Serialize and Deserialize; use #[serde(skip)]",
                )));
            }
            if attrs.skip_serializing() {
                return None;
            }
            if attrs.flatten() {
                return Some(Err(Error::new_spanned(
                    field.original,
                    "UzuToolSchema does not support #[serde(flatten)]",
                )));
            }
            if attrs.skip_serializing_if().is_some() {
                return Some(Err(Error::new_spanned(
                    field.original,
                    "UzuToolSchema does not support #[serde(skip_serializing_if = \"...\")]",
                )));
            }
            if attrs.serialize_with().is_some() || attrs.deserialize_with().is_some() || attrs.getter().is_some() {
                return Some(Err(Error::new_spanned(
                    field.original,
                    "UzuToolSchema does not support Serde field attributes that change the serialized representation",
                )));
            }

            let serialize_name = attrs.name().serialize_name();
            let deserialize_name = attrs.name().deserialize_name();
            if serialize_name != deserialize_name {
                return Some(Err(Error::new_spanned(
                    field.original,
                    "UzuToolSchema requires matching Serialize and Deserialize field names",
                )));
            }

            let required =
                !is_option(field.ty) && attrs.default().is_none() && serde_container.attrs.default().is_none();
            Some(Ok((field.original, serialize_name.to_string(), required)))
        })
        .collect::<syn::Result<Vec<_>>>()?;

    let ident = &input.ident;
    let mut generics = input.generics.clone();
    for (field, _, _) in &schema_fields {
        let ty = &field.ty;
        generics.make_where_clause().predicates.push(syn::parse_quote!(#ty: #nagare::tool::schema::ToolSchema));
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let schema = if schema_fields.is_empty() {
        quote!(#nagare::tool::schema::JsonSchema::empty_object())
    } else {
        let properties = schema_fields.iter().map(|(field, name, _)| {
            let ty = &field.ty;
            let schema = quote!(<#ty as #nagare::tool::schema::ToolSchema>::json_schema());
            let description = doc_string(&field.attrs);
            if description.is_empty() {
                quote!((#name, #schema))
            } else {
                quote!((#name, #schema.with_description(#description)))
            }
        });
        let required = schema_fields.iter().filter(|(_, _, required)| *required).map(|(_, name, _)| name);
        quote! {
            #nagare::tool::schema::JsonSchema::object(
                [#(#properties),*],
                ::std::vec::Vec::<&str>::from([#(#required),*]),
            )
        }
    };
    let with_additional_properties = if serde_container.attrs.deny_unknown_fields() {
        quote!(.with_additional_properties(false))
    } else {
        quote!()
    };

    let struct_description = doc_string(&input.attrs);
    let with_description = if struct_description.is_empty() {
        quote!()
    } else {
        quote!(.with_description(#struct_description))
    };

    Ok(quote! {
        impl #impl_generics #nagare::tool::schema::ToolSchema for #ident #ty_generics #where_clause {
            fn json_schema() -> #nagare::tool::schema::JsonSchema {
                #schema #with_additional_properties #with_description
            }
        }
    })
}
