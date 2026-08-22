use reqwest::Url;

use super::is_https_downgrade;

#[test]
fn redirect_policy_rejects_downgrades() {
    let previous = Url::parse("https://huggingface.co/model").unwrap();
    let downgrade = Url::parse("http://cdn.example.test/model").unwrap();
    let safe = Url::parse("https://cdn.example.test/model").unwrap();

    assert!(is_https_downgrade(Some(&previous), &downgrade));
    assert!(!is_https_downgrade(Some(&previous), &safe));
}
