use syn::{Token, Variant, punctuated::Punctuated};

pub enum ErrorShape {
    Unit,
    Data,
}

impl ErrorShape {
    pub fn from_variants(variants: &Punctuated<Variant, Token![,]>) -> Self {
        if variants.iter().all(|variant| matches!(variant.fields, syn::Fields::Unit)) {
            ErrorShape::Unit
        } else {
            ErrorShape::Data
        }
    }
}

pub struct ErrorContext<'item> {
    pub item: &'item syn::ItemEnum,
    pub shape: ErrorShape,
}

impl<'item> ErrorContext<'item> {
    pub fn from_item(item: &'item syn::ItemEnum) -> Self {
        let shape = ErrorShape::from_variants(&item.variants);
        Self {
            item,
            shape,
        }
    }
}
