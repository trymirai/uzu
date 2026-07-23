#![cfg(all(feature = "metal", target_os = "macos"))]

use itertools::Itertools;

pub fn unqualify_variant(value: &str) -> &str {
    value.rsplit("::").next().unwrap_or(value)
}

pub fn static_mangle(
    function_name: impl AsRef<str>,
    variant: impl IntoIterator<Item = impl AsRef<str>>,
) -> String {
    format!(
        "_D{}{}{}",
        function_name.as_ref().len(),
        function_name.as_ref(),
        variant
            .into_iter()
            .map(|v| {
                let v = unqualify_variant(v.as_ref()).replace('-', "n");
                format!("S{}V{}", v.len(), v)
            })
            .join("")
    )
}
