use proc_macro2::TokenStream;

pub fn get_generic_name_stream(var_name: &str) -> TokenStream {
    let mut generic_name: String = var_name
        .split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect();
    generic_name.insert(0, 'T');
    generic_name.parse().unwrap()
}
