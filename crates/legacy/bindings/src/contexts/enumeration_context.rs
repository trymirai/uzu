use syn::{Token, Variant, punctuated::Punctuated};

pub enum EnumerationShape {
    Unit,
    Data,
}

impl EnumerationShape {
    pub fn from_variants(variants: &Punctuated<Variant, Token![,]>) -> syn::Result<Self> {
        let mut has_unit_variant = false;
        let mut has_named_variant = false;
        for variant in variants {
            match variant.fields {
                syn::Fields::Unit => has_unit_variant = true,
                syn::Fields::Named(_) => has_named_variant = true,
                syn::Fields::Unnamed(_) => {
                    return Err(syn::Error::new_spanned(
                        &variant.fields,
                        "Bindings::export(Enumeration) variants must use named fields (use `{}` for empty variants), tuple variants are not supported",
                    ));
                },
            }
        }
        if has_unit_variant && has_named_variant {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "Bindings::export(Enumeration) variants must all be unit (e.g. `Foo`) or all be named (e.g. `Foo {}`, `Foo { x: i64 }`), mixing is not supported",
            ));
        }
        if has_named_variant {
            Ok(EnumerationShape::Data)
        } else {
            Ok(EnumerationShape::Unit)
        }
    }
}

pub struct EnumerationContext<'item> {
    pub item: &'item syn::ItemEnum,
    pub shape: EnumerationShape,
}

impl<'item> EnumerationContext<'item> {
    pub fn from_item(item: &'item syn::ItemEnum) -> syn::Result<Self> {
        let shape = EnumerationShape::from_variants(&item.variants)?;
        Ok(Self {
            item,
            shape,
        })
    }
}
