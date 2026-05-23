use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{Data, DeriveInput, Field, Generics, parse_macro_input, parse_quote};

pub fn uzu_config(
    _args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    let mut input = parse_macro_input!(input as DeriveInput);

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
