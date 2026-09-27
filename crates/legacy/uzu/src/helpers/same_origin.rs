use reqwest::Url;

pub fn same_origin(
    url: &str,
    endpoint: &str,
) -> bool {
    match (Url::parse(url), Url::parse(endpoint)) {
        (Ok(url), Ok(endpoint)) => url.origin() == endpoint.origin(),
        _ => false,
    }
}
