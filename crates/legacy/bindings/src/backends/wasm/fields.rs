use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::ext::IdentExt;

use super::typescript::{cfg_attributes, class_member, custom_section, js_name, object_member};
use crate::backends::typescript::{option_inner, typescript_type};

pub struct Field<'item> {
    pub ident: &'item syn::Ident,
    pub ty: &'item syn::Type,
    pub attributes: &'item [syn::Attribute],
    /// The JavaScript name: camelCase for named fields, `field0` for tuple fields as napi does.
    pub js_name: String,
    pub parameter: syn::Ident,
}

pub fn named_fields(fields: &syn::Fields) -> Vec<Field<'_>> {
    fields
        .iter()
        .filter_map(|field| {
            let ident = field.ident.as_ref()?;
            let js_name = js_name(&ident.unraw().to_string());
            Some(Field {
                ident,
                ty: &field.ty,
                attributes: &field.attrs,
                parameter: format_ident!("{}", js_name),
                js_name,
            })
        })
        .collect()
}

/// Fields of a class can't be conditionally compiled: the constructor signature would change.
pub fn reject_cfg_fields(fields: &[Field]) -> Option<TokenStream> {
    let field = fields.iter().find(|field| !cfg_attributes(field.attributes).is_empty())?;
    Some(
        syn::Error::new_spanned(field.ident, "bindings::export does not support #[cfg] on fields of wasm classes")
            .to_compile_error(),
    )
}

/// Constructor parameters in field order, taking JavaScript values. A trailing run of `Option`
/// fields becomes optional parameters, as in napi.
pub fn constructor_parameters(
    fields: &[Field],
    self_name: &str,
) -> Vec<TokenStream> {
    let optional_from =
        fields.iter().rposition(|field| option_inner(field.ty).is_none()).map_or(0, |position| position + 1);
    fields
        .iter()
        .enumerate()
        .map(|(index, field)| {
            let parameter = &field.parameter;
            match option_inner(field.ty) {
                Some(inner) if index >= optional_from => {
                    let ty = format!("{} | null", typescript_type(inner, self_name));
                    quote! { #[wasm_bindgen(unchecked_optional_param_type = #ty)] #parameter: ::wasm_bindgen::JsValue }
                },
                Some(inner) => {
                    let ty = format!("{} | undefined | null", typescript_type(inner, self_name));
                    quote! { #[wasm_bindgen(unchecked_param_type = #ty)] #parameter: ::wasm_bindgen::JsValue }
                },
                None => {
                    let ty = typescript_type(field.ty, self_name);
                    quote! { #[wasm_bindgen(unchecked_param_type = #ty)] #parameter: ::wasm_bindgen::JsValue }
                },
            }
        })
        .collect()
}

/// `field: FromJs::from_js(parameter)?` initializers for the constructor parameters.
pub fn field_initializers(fields: &[Field]) -> Vec<TokenStream> {
    fields
        .iter()
        .map(|field| {
            let ident = field.ident;
            let parameter = &field.parameter;
            quote! { #ident: ::bindings_types::wasm::FromJs::from_js(#parameter)? }
        })
        .collect()
}

/// A getter and setter for each field. `read` and `write` produce the expression for the field's
/// value and the statement that assigns `__wasm_value` to it, so variant classes can reach into their
/// enum (whose fields may themselves be called `value`).
pub fn accessors(
    fields: &[Field],
    read: impl Fn(&Field) -> TokenStream,
    write: impl Fn(&Field) -> TokenStream,
) -> Vec<TokenStream> {
    fields
        .iter()
        .map(|field| {
            let js_name = &field.js_name;
            let field_ty = field.ty;
            let raw = field.ident.unraw();
            let getter = format_ident!("__wasm_get_{}", raw);
            let setter = format_ident!("__wasm_set_{}", raw);
            let read = read(field);
            let write = write(field);
            // Declared by `field_declarations` instead: wasm-bindgen can't mark a property optional.
            quote! {
                #[wasm_bindgen(getter = #js_name, skip_typescript)]
                pub fn #getter(&self) -> ::wasm_bindgen::JsValue {
                    ::bindings_types::wasm::IntoJs::into_js(::core::clone::Clone::clone(#read))
                }

                #[wasm_bindgen(setter = #js_name, skip_typescript)]
                pub fn #setter(&mut self, value: ::wasm_bindgen::JsValue) -> ::bindings_types::wasm::JsResult<()> {
                    let __wasm_value = <#field_ty as ::bindings_types::wasm::FromJs>::from_js(value)?;
                    #write
                    Ok(())
                }
            }
        })
        .collect()
}

/// Statements setting each field of `source` (an expression path such as `self.` or a binding
/// prefix) on `object`. `None` fields are left out, as napi does for objects.
pub fn object_setters(
    fields: &[Field],
    value: impl Fn(&Field) -> TokenStream,
) -> Vec<TokenStream> {
    fields
        .iter()
        .map(|field| {
            let js_name = &field.js_name;
            let cfgs = cfg_attributes(field.attributes);
            let value = value(field);
            if option_inner(field.ty).is_some() {
                quote! {
                    #(#cfgs)*
                    if let Some(value) = #value {
                        ::bindings_types::wasm::set_property(&object, #js_name, ::bindings_types::wasm::IntoJs::into_js(value));
                    }
                }
            } else {
                quote! {
                    #(#cfgs)*
                    ::bindings_types::wasm::set_property(&object, #js_name, ::bindings_types::wasm::IntoJs::into_js(#value));
                }
            }
        })
        .collect()
}

/// `field: FromJs::from_js(value.field)?` initializers reading the fields of the object `value`.
pub fn object_initializers(fields: &[Field]) -> Vec<TokenStream> {
    fields
        .iter()
        .map(|field| {
            let ident = field.ident;
            let js_name = &field.js_name;
            let cfgs = cfg_attributes(field.attributes);
            quote! {
                #(#cfgs)*
                #ident: ::bindings_types::wasm::FromJs::from_js(::bindings_types::wasm::get_property(&value, #js_name)?)?
            }
        })
        .collect()
}

/// The members of an object type, e.g. `name: string; description?: string`.
pub fn object_members(fields: &[Field]) -> Vec<String> {
    fields.iter().map(|field| object_member(&field.js_name, field.ty)).collect()
}

/// Declares the fields of a class as napi does (`name?: T | null` for an `Option`), in an interface
/// that merges with the class wasm-bindgen declares.
pub fn field_declarations(
    class_name: &str,
    cfgs: &[&syn::Attribute],
    fields: &[Field],
) -> TokenStream {
    let members: String =
        fields.iter().map(|field| format!("  {};\n", class_member(&field.js_name, field.ty))).collect();
    let declaration = format!("export interface {class_name} {{\n{members}}}");
    custom_section(&format!("{class_name}Fields"), cfgs, &declaration)
}
