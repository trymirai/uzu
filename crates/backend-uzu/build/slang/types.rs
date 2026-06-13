use std::{borrow::Cow, collections::HashMap};

use itertools::Itertools;
use regex::Regex;

pub fn rust2slang(ty: &str) -> Option<&'static str> {
    match ty {
        "bool" => Some("bool"),
        "i8" => Some("int8_t"),
        "u8" => Some("uint8_t"),
        "i16" => Some("int16_t"),
        "u16" => Some("uint16_t"),
        "i32" => Some("int"),
        "u32" => Some("uint"),
        "i64" => Some("int64_t"),
        "u64" => Some("uint64_t"),
        "f16" => Some("half"),
        "f32" => Some("float"),
        "f64" => Some("double"),
        _ => None,
    }
}

pub fn slang2rust<'a>(
    ty: &str,
    gpu_type_map: &'a HashMap<String, String>,
) -> Option<Cow<'a, str>> {
    match ty {
        "bool" => Some(Cow::Borrowed("bool")),
        "int8_t" => Some(Cow::Borrowed("i8")),
        "uint8_t" => Some(Cow::Borrowed("u8")),
        "int16_t" => Some(Cow::Borrowed("i16")),
        "uint16_t" => Some(Cow::Borrowed("u16")),
        "int" => Some(Cow::Borrowed("i32")),
        "uint" => Some(Cow::Borrowed("u32")),
        "int64_t" => Some(Cow::Borrowed("i64")),
        "uint64_t" => Some(Cow::Borrowed("u64")),
        "half" => Some(Cow::Borrowed("f16")),
        "float" => Some(Cow::Borrowed("f32")),
        "double" => Some(Cow::Borrowed("f64")),
        _ if let Some(ty_translated) = gpu_type_map.get(ty) => Some(Cow::Borrowed(ty_translated)),
        _ => {
            let (scalar_ty, size) = ty.rsplit_once('[')?;
            let size = size.strip_suffix(']')?;
            let elem = slang2rust(scalar_ty, gpu_type_map)?;
            Some(Cow::Owned(if size.is_empty() {
                format!("&[{elem}]")
            } else {
                let size: usize = size.parse().ok()?;
                format!("&[{elem}; {size}]")
            }))
        },
    }
}

pub struct Specializer<'a> {
    regex: Regex,
    map: &'a [(&'a str, &'a str)],
}

impl<'a> Specializer<'a> {
    pub fn new(map: &'a [(&'a str, &'a str)]) -> Self {
        let regex =
            Regex::new(&format!(r"\b(?:{})\b", map.iter().map(|(key, _value)| regex::escape(key)).join("|"))).unwrap();

        Self {
            regex,
            map,
        }
    }

    pub fn specialize<'b>(
        &self,
        generic_type: &'b str,
    ) -> Cow<'b, str> {
        if self.map.is_empty() {
            return Cow::Borrowed(generic_type);
        }
        self.regex.replace_all(generic_type, |c: &regex::Captures| {
            self.map.iter().find_map(|(key, value)| (*key == &c[0]).then_some(*value)).unwrap()
        })
    }
}
