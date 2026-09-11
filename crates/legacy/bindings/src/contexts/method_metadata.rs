use syn::Ident;

use crate::types::MethodFlavor;

pub struct MethodMetadata {
    pub method: syn::ImplItemFn,
    pub flavor: MethodFlavor,
}

impl MethodMetadata {
    pub fn is_async(&self) -> bool {
        self.method.sig.asyncness.is_some()
    }

    pub fn returns_result(&self) -> bool {
        let return_type = match &self.method.sig.output {
            syn::ReturnType::Default => return false,
            syn::ReturnType::Type(_, return_type) => return_type.as_ref(),
        };
        let path = match return_type {
            syn::Type::Path(type_path) => &type_path.path,
            _ => return false,
        };
        path.segments.last().map(|segment| segment.ident == "Result").unwrap_or(false)
    }

    pub fn receiver(&self) -> Option<&syn::Receiver> {
        self.method.sig.inputs.iter().find_map(|argument| match argument {
            syn::FnArg::Receiver(receiver) => Some(receiver),
            _ => None,
        })
    }

    pub fn typed_args(&self) -> Vec<&syn::PatType> {
        self.method
            .sig
            .inputs
            .iter()
            .filter_map(|argument| match argument {
                syn::FnArg::Typed(pat_type) => Some(pat_type),
                syn::FnArg::Receiver(_) => None,
            })
            .collect()
    }

    pub fn arg_idents(&self) -> Vec<Ident> {
        self.typed_args()
            .iter()
            .filter_map(|pat_type| match pat_type.pat.as_ref() {
                syn::Pat::Ident(pat_ident) => Some(pat_ident.ident.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn callback_inputs(&self) -> Option<Vec<syn::Type>> {
        for input in &self.method.sig.inputs {
            let syn::FnArg::Typed(pat_type) = input else {
                continue;
            };
            if let Some(inputs) = extract_void_fn_inputs(&pat_type.ty) {
                return Some(inputs);
            }
        }
        None
    }
}

fn extract_void_fn_inputs(ty: &syn::Type) -> Option<Vec<syn::Type>> {
    let inner = extract_box_inner(ty)?;
    let syn::Type::TraitObject(trait_object) = inner else {
        return None;
    };
    for bound in &trait_object.bounds {
        let syn::TypeParamBound::Trait(trait_bound) = bound else {
            continue;
        };
        let segment = trait_bound.path.segments.last()?;
        if segment.ident != "Fn" && segment.ident != "FnMut" && segment.ident != "FnOnce" {
            continue;
        }
        let syn::PathArguments::Parenthesized(paren) = &segment.arguments else {
            continue;
        };
        let returns_unit = match &paren.output {
            syn::ReturnType::Default => true,
            syn::ReturnType::Type(_, ty) => match ty.as_ref() {
                syn::Type::Tuple(tuple) => tuple.elems.is_empty(),
                _ => false,
            },
        };
        if !returns_unit {
            return None;
        }
        return Some(paren.inputs.iter().cloned().collect());
    }
    None
}

fn extract_box_inner(ty: &syn::Type) -> Option<&syn::Type> {
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != "Box" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    for argument in &args.args {
        if let syn::GenericArgument::Type(inner) = argument {
            return Some(inner);
        }
    }
    None
}
