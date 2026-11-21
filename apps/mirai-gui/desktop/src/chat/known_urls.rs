use std::collections::HashMap;

use reqwest::Url;

pub(super) const MAX_URL_BYTES: usize = 2_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Origin {
    User,
    Trusted,
    Page {
        host: String,
    },
}

struct Known {
    page_host: Option<String>,
    shown_as_http: bool,
    written_by_user: bool,
}

#[derive(Default)]
pub(super) struct KnownUrls(HashMap<String, Known>);

impl KnownUrls {
    pub(super) fn forget(&mut self) {
        self.0.clear();
    }

    pub(super) fn remember(
        &mut self,
        url: &Url,
        origin: Origin,
    ) {
        let written_by_user = origin == Origin::User;
        let page_host = match origin {
            Origin::Page {
                host,
            } => Some(host),
            Origin::User | Origin::Trusted => None,
        };
        let known = self.0.entry(page_key(url)).or_insert_with(|| Known {
            page_host: page_host.clone(),
            shown_as_http: false,
            written_by_user: false,
        });
        if page_host.is_none() {
            known.page_host = None;
        }
        known.shown_as_http |= url.scheme() == "http";
        known.written_by_user |= written_by_user;
    }

    pub(super) fn remember_text(
        &mut self,
        text: &str,
        origin: &Origin,
    ) {
        let loose = (*origin == Origin::User).then(|| loosely_written_urls_in(text));
        for url in urls_in(text).chain(loose.into_iter().flatten()) {
            self.remember(&url, origin.clone());
        }
    }

    pub(super) fn remember_addresses<'a>(
        &mut self,
        addresses: impl IntoIterator<Item = &'a str>,
        origin: &Origin,
    ) {
        for url in addresses.into_iter().filter_map(|address| Url::parse(address).ok()) {
            self.remember(&url, origin.clone());
        }
    }

    pub(super) fn is_openable(
        &self,
        url: &Url,
    ) -> bool {
        self.0.get(&page_key(url)).is_some_and(|known| match &known.page_host {
            Some(host) => url.query().is_none() || url.host_str().is_some_and(|own| own.eq_ignore_ascii_case(host)),
            None => true,
        })
    }

    pub(super) fn was_written_by_user(
        &self,
        url: &Url,
    ) -> bool {
        self.0.get(&page_key(url)).is_some_and(|known| known.written_by_user)
    }

    pub(super) fn was_shown_as_http(
        &self,
        url: &Url,
    ) -> bool {
        self.0.get(&page_key(url)).is_some_and(|known| known.shown_as_http)
    }
}

fn page_key(url: &Url) -> String {
    let host = url.host_str().unwrap_or_default().to_lowercase();
    let port = url.port().map(|port| format!(":{port}")).unwrap_or_default();
    let path = url.path().trim_end_matches('/');
    let query = url.query().map(|query| format!("?{query}")).unwrap_or_default();
    format!("{host}{port}{path}{query}")
}

fn urls_in(text: &str) -> impl Iterator<Item = Url> + '_ {
    text.match_indices("http").filter_map(move |(start, _)| {
        let rest = &text[start..];
        if !(rest.starts_with("http://") || rest.starts_with("https://")) {
            return None;
        }
        let end = rest
            .char_indices()
            .take_while(|(index, _)| *index <= MAX_URL_BYTES)
            .find_map(|(index, character)| is_address_end(character).then_some(index))
            .or((rest.len() <= MAX_URL_BYTES).then_some(rest.len()))?;
        Url::parse(without_trailing_punctuation(&rest[..end])).ok().filter(|url| url.host_str().is_some())
    })
}

fn is_address_end(character: char) -> bool {
    character.is_whitespace() || matches!(character, '<' | '>' | '"' | '\'' | '`')
}

fn without_trailing_punctuation(mut candidate: &str) -> &str {
    let mut unbalanced = candidate.matches(')').count().saturating_sub(candidate.matches('(').count());
    loop {
        candidate = candidate.trim_end_matches(['.', ',', ';', ':', '!', '?', '*']);
        if unbalanced == 0 || !candidate.ends_with(')') {
            return candidate;
        }
        candidate = &candidate[..candidate.len() - 1];
        unbalanced -= 1;
    }
}

fn loosely_written_urls_in(text: &str) -> impl Iterator<Item = Url> + '_ {
    text.split(is_address_end).filter(|word| word.len() <= MAX_URL_BYTES).filter_map(|word| {
        let word = without_trailing_punctuation(word.trim_start_matches(['(', '[', '*', '_']));
        if let Some((scheme, _)) = word.split_once("://") {
            let is_web = scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https");
            return is_web.then(|| Url::parse(word).ok()).flatten().filter(|url| url.host_str().is_some());
        }
        let host = word.split(['/', '?', '#']).next().unwrap_or_default();
        is_host_name(host).then(|| Url::parse(&format!("https://{word}")).ok()).flatten()
    })
}

fn is_host_name(host: &str) -> bool {
    let Some((others, last)) = host.rsplit_once('.') else {
        return false;
    };
    others.split('.').all(|label| !label.is_empty() && label.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'))
        && last.len() >= 2
        && last.chars().all(|c| c.is_ascii_alphabetic())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(text: &str) -> Url {
        Url::parse(text).unwrap()
    }

    fn found(text: &str) -> Vec<String> {
        urls_in(text).map(String::from).collect()
    }

    fn from_page(host: &str) -> Origin {
        Origin::Page {
            host: host.to_string(),
        }
    }

    #[test]
    fn addresses_are_found_in_prose_and_markdown_with_their_punctuation_left_behind() {
        assert_eq!(
            found("See https://example.com/a, then (https://example.com/b). Also <https://example.com/c>!"),
            ["https://example.com/a", "https://example.com/b", "https://example.com/c"]
        );
        assert_eq!(
            found("[Rust](https://en.wikipedia.org/wiki/Rust_(programming_language)) and **https://example.com/bold**"),
            ["https://en.wikipedia.org/wiki/Rust_(programming_language)", "https://example.com/bold"]
        );
        assert_eq!(found("http://example.com/plain?q=1&r=2#part"), ["http://example.com/plain?q=1&r=2#part"]);
        assert!(found("https:// nothing, httpx://no, and http:/one-slash").is_empty());
    }

    #[test]
    fn an_address_the_user_typed_without_a_scheme_is_read_as_https() {
        let loose = |text: &str| loosely_written_urls_in(text).map(String::from).collect::<Vec<_>>();
        assert_eq!(
            loose("Summarise example.com/pricing, then www.example.org. Also (docs.example.net/a_(b))!"),
            ["https://example.com/pricing", "https://www.example.org/", "https://docs.example.net/a_(b)"]
        );
        assert_eq!(
            loose("HTTPS://Example.com/Path and Http://plain.example"),
            ["https://example.com/Path", "http://plain.example/"]
        );
        assert_eq!(loose("open main.rs"), ["https://main.rs/"]);
        for not_a_page in [
            "mail me at name@example.com",
            "version v1.2 or 3.14",
            "e.g. this, i.e. that, the U.S. and so on...",
            "ftp://example.com/file and localhost:3000/x",
            "one.two.3 and a..b and .com and example.",
        ] {
            assert!(loose(not_a_page).is_empty(), "{not_a_page}");
        }
    }

    #[test]
    fn only_the_users_own_words_are_read_that_loosely() {
        let mut known = KnownUrls::default();
        known.remember_text("Read example.com/pricing", &Origin::Trusted);
        known.remember_text("Read example.com/pricing", &from_page("news.example"));
        assert!(!known.is_openable(&url("https://example.com/pricing")));

        known.remember_text("Read example.com/pricing", &Origin::User);
        assert!(known.is_openable(&url("https://example.com/pricing")));
        assert!(known.was_written_by_user(&url("https://example.com/pricing")));
        assert!(!known.was_shown_as_http(&url("http://example.com/pricing")));
    }

    #[test]
    fn one_page_spelled_several_ways_is_one_key() {
        let base = page_key(&url("https://Example.com/path/"));
        for same in ["http://example.com/path", "https://example.com/path#section", "HTTPS://EXAMPLE.com/path/"] {
            assert_eq!(page_key(&url(same)), base, "{same}");
        }
        for other in [
            "https://example.com/path?x=1",
            "https://example.com/other",
            "https://sub.example.com/path",
            "https://example.com:8443/path",
        ] {
            assert_ne!(page_key(&url(other)), base, "{other}");
        }
    }

    #[test]
    fn only_remembered_addresses_are_known_until_the_conversation_is_forgotten() {
        let mut known = KnownUrls::default();
        assert!(!known.is_openable(&url("https://example.com/seen")));

        known.remember_text("Read https://example.com/seen and [this](https://example.com/linked).", &Origin::Trusted);
        known.remember_addresses(["https://example.com/result?q=1", "not an address"], &Origin::Trusted);
        for openable in ["https://example.com/seen", "http://example.com/linked/", "https://example.com/result?q=1#top"]
        {
            assert!(known.is_openable(&url(openable)), "{openable}");
        }
        assert!(!known.is_openable(&url("https://example.com/made-up")));

        known.forget();
        assert!(!known.is_openable(&url("https://example.com/seen")));
    }

    #[test]
    fn a_link_from_a_page_carries_no_query_to_another_host() {
        let mut known = KnownUrls::default();
        known.remember_text(
            "More at https://news.example/story?id=7 and https://Other.example/collect?secret=SEND_THIS \
             or plain https://other.example/about",
            &from_page("news.example"),
        );
        assert!(known.is_openable(&url("https://news.example/story?id=7")), "own host keeps its query");
        assert!(known.is_openable(&url("https://other.example/about")), "another host, no query");
        assert!(
            !known.is_openable(&url("https://other.example/collect?secret=SEND_THIS")),
            "another host with a query"
        );

        known.remember(&url("https://other.example/collect?secret=SEND_THIS"), Origin::Trusted);
        assert!(known.is_openable(&url("https://other.example/collect?secret=SEND_THIS")));
        known.remember(&url("https://other.example/collect?secret=SEND_THIS"), from_page("news.example"));
        assert!(known.is_openable(&url("https://other.example/collect?secret=SEND_THIS")));
    }

    #[test]
    fn a_page_is_read_in_the_clear_only_when_it_was_shown_that_way() {
        let mut known = KnownUrls::default();
        known.remember_text("See https://secure.example/a and http://plain.example/b", &Origin::Trusted);
        assert!(known.is_openable(&url("http://secure.example/a")), "one page, whatever the scheme");
        assert!(!known.was_shown_as_http(&url("http://secure.example/a")));
        assert!(known.was_shown_as_http(&url("http://plain.example/b")));
        known.forget();
        assert!(!known.was_shown_as_http(&url("http://plain.example/b")));
    }

    #[test]
    fn an_endless_address_or_run_of_text_yields_next_to_nothing() {
        assert!(found(&format!("https://example.com/{}", "a".repeat(MAX_URL_BYTES))).is_empty());
        assert!(found(&"http://".repeat(20_000)).len() <= MAX_URL_BYTES / "http://".len() + 1);
    }
}
