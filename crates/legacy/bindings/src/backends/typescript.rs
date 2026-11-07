//! The TypeScript types napi declares for Rust types, shared by the napi and wasm backends so both
//! packages declare the same API.

/// The TypeScript type napi declares for a Rust type. `Option` maps to `T | null`, as for return
/// values; fields and parameters handle a top-level `Option` themselves.
pub fn typescript_type(
    ty: &syn::Type,
    self_name: &str,
) -> String {
    match ty {
        syn::Type::Reference(reference) => typescript_type(&reference.elem, self_name),
        syn::Type::Paren(paren) => typescript_type(&paren.elem, self_name),
        syn::Type::Group(group) => typescript_type(&group.elem, self_name),
        syn::Type::Tuple(tuple) if tuple.elems.is_empty() => "void".to_string(),
        syn::Type::Slice(slice) => format!("Array<{}>", typescript_type(&slice.elem, self_name)),
        syn::Type::Path(path) => path_type(path, self_name),
        _ => "any".to_string(),
    }
}

fn path_type(
    path: &syn::TypePath,
    self_name: &str,
) -> String {
    let Some(segment) = path.path.segments.last() else {
        return "any".to_string();
    };
    let arguments = generic_arguments(segment);
    let argument = |index: usize| {
        arguments.get(index).map_or_else(|| "any".to_string(), |argument| typescript_type(argument, self_name))
    };
    match segment.ident.to_string().as_str() {
        "bool" => "boolean".to_string(),
        "u8" | "u16" | "u32" | "u64" | "usize" | "i8" | "i16" | "i32" | "i64" | "isize" | "f32" | "f64" => {
            "number".to_string()
        },
        "String" | "str" | "char" => "string".to_string(),
        "Option" => format!("{} | null", argument(0)),
        "Vec" | "VecDeque" => format!("Array<{}>", argument(0)),
        "HashMap" | "BTreeMap" | "IndexMap" => format!("Record<{}, {}>", argument(0), argument(1)),
        "Box" | "Arc" | "Rc" | "Result" => argument(0),
        "JsValue" => "any".to_string(),
        "Self" => self_name.to_string(),
        name => name.to_string(),
    }
}

/// The type of a class field `name?: T | null`, for an `Option` field: its getter returns `null`
/// for `None` in both packages, so the type includes `null` as well as the `?` napi gives it.
pub fn optional_class_field_type(
    inner: &syn::Type,
    self_name: &str,
) -> String {
    format!("{} | null", typescript_type(inner, self_name))
}

fn generic_arguments(segment: &syn::PathSegment) -> Vec<&syn::Type> {
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return Vec::new();
    };
    arguments
        .args
        .iter()
        .filter_map(|argument| match argument {
            syn::GenericArgument::Type(ty) => Some(ty),
            _ => None,
        })
        .collect()
}

fn wrapped_inner<'ty>(
    ty: &'ty syn::Type,
    wrapper: &str,
) -> Option<&'ty syn::Type> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    let segment = path.path.segments.last()?;
    if segment.ident != wrapper {
        return None;
    }
    generic_arguments(segment).into_iter().next()
}

pub fn option_inner(ty: &syn::Type) -> Option<&syn::Type> {
    wrapped_inner(ty, "Option")
}

pub fn result_inner(ty: &syn::Type) -> Option<&syn::Type> {
    wrapped_inner(ty, "Result")
}

pub fn is_unit(ty: &syn::Type) -> bool {
    matches!(ty, syn::Type::Tuple(tuple) if tuple.elems.is_empty())
}
