use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};

use reqwest::{
    Url,
    dns::{Addrs, Name, Resolve, Resolving},
};

#[derive(Debug, PartialEq, Eq)]
pub(super) struct NotPublic;

impl std::fmt::Display for NotPublic {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.write_str("the address is not that of a public web page")
    }
}

impl std::error::Error for NotPublic {}

pub(super) fn public_web_url(address: &str) -> Result<Url, NotPublic> {
    Url::parse(address.trim()).ok().filter(is_public_web).ok_or(NotPublic)
}

pub(super) fn is_public_web(url: &Url) -> bool {
    let is_web = matches!(url.scheme(), "http" | "https");
    let has_credentials = !url.username().is_empty() || url.password().is_some();
    let host_is_public = match literal_address(url) {
        Some(literal) => is_public(literal),
        None => url.host_str().is_some_and(|host| !host.is_empty()),
    };
    is_web && !has_credentials && host_is_public
}

fn literal_address(url: &Url) -> Option<IpAddr> {
    url.host_str()?.trim_start_matches('[').trim_end_matches(']').parse().ok()
}

pub(super) fn resolves_to_public(url: &Url) -> bool {
    resolution(url) == Resolution::Public
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Resolution {
    Public,
    NotPublic,
    Failed,
}

pub(super) fn resolution(url: &Url) -> Resolution {
    let public = |yes: bool| {
        if yes {
            Resolution::Public
        } else {
            Resolution::NotPublic
        }
    };
    if let Some(literal) = literal_address(url) {
        return public(is_public(literal));
    }
    let Some(host) = url.host_str() else {
        return Resolution::NotPublic;
    };
    match (host, 0).to_socket_addrs() {
        Ok(resolved) => {
            let addresses: Vec<SocketAddr> = resolved.collect();
            public(!addresses.is_empty() && addresses.iter().all(|address| is_public(address.ip())))
        },
        Err(_) => Resolution::Failed,
    }
}

fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => is_public_v4(ip),
        IpAddr::V6(ip) => match ip.to_ipv4_mapped() {
            Some(mapped) => is_public_v4(mapped),
            None => is_public_v6(ip),
        },
    }
}

fn is_public_v4(ip: Ipv4Addr) -> bool {
    let [first, second, third, _] = ip.octets();
    let carrier_nat = first == 100 && (64..128).contains(&second);
    let protocol_assignments = first == 192 && second == 0 && third == 0;
    let benchmarking = first == 198 && (second == 18 || second == 19);
    let reserved = first >= 240;
    !(ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_multicast()
        || ip.is_documentation()
        || carrier_nat
        || protocol_assignments
        || benchmarking
        || reserved
        || first == 0)
}

fn is_public_v6(ip: Ipv6Addr) -> bool {
    let segments = ip.segments();
    let first = segments[0];
    let unique_local = first & 0xfe00 == 0xfc00;
    let link_local = first & 0xffc0 == 0xfe80;
    let teredo = first == 0x2001 && segments[1] == 0;
    // Translation and tunnel prefixes carry an IPv4 address, which is judged on its own.
    let embedded_v4 = |high: u16, low: u16| is_public_v4(Ipv4Addr::from((u32::from(high) << 16) | u32::from(low)));
    let nat64 = first == 0x64 && segments[1] == 0xff9b && segments[2..6] == [0, 0, 0, 0];
    let local_nat64 = first == 0x64 && segments[1] == 0xff9b && segments[2] == 1;
    let six_to_four = first == 0x2002;
    let v4_compatible = segments[..6] == [0, 0, 0, 0, 0, 0];
    let site_local = first & 0xffc0 == 0xfec0;
    if nat64 || v4_compatible {
        return embedded_v4(segments[6], segments[7]);
    }
    if six_to_four {
        return embedded_v4(segments[1], segments[2]);
    }
    !(ip.is_multicast() || unique_local || link_local || site_local || teredo || local_nat64)
}

pub(super) struct PublicAddresses;

impl Resolve for PublicAddresses {
    fn resolve(
        &self,
        name: Name,
    ) -> Resolving {
        Box::pin(async move {
            let host = name.as_str().to_string();
            let resolved = tokio::task::spawn_blocking(move || (host.as_str(), 0).to_socket_addrs()).await??;
            let public: Vec<SocketAddr> = resolved.filter(|address| is_public(address.ip())).collect();
            if public.is_empty() {
                return Err("the host has no public address".into());
            }
            Ok(Box::new(public.into_iter()) as Addrs)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_public_web_addresses_are_accepted() {
        for address in
            ["https://example.com/a?b=1", " http://example.com ", "https://8.8.8.8/", "https://[2606:4700::1111]/"]
        {
            assert!(public_web_url(address).is_ok(), "{address}");
        }
        assert!(public_web_url("http://[64:ff9b::808:808]/").is_ok());
        assert!(public_web_url("http://[2002:808:808::]/").is_ok());
        for address in [
            "example.com",
            "ftp://example.com/",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "https://user:secret@example.com/",
            "http://127.0.0.1:3002/",
            "http://2130706433/",
            "http://0x7f.1/",
            "http://0.0.0.0/",
            "http://10.0.0.5/",
            "http://192.168.1.1/",
            "http://169.254.169.254/latest/meta-data",
            "http://100.64.0.1/",
            "http://[::1]/",
            "http://[fe80::1]/",
            "http://[fd00::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[64:ff9b::a00:1]/",
            "http://[2002:7f00:1::]/",
            "http://[2001:0:4136:e378::1]/",
            "http://[::7f00:1]/",
            "http://[::]/",
            "http://[64:ff9b:1::a00:1]/",
            "http://[fec0::1]/",
            "http://192.0.0.9/",
            "http://198.18.0.1/",
            "http://240.0.0.1/",
        ] {
            assert_eq!(public_web_url(address).unwrap_err(), NotPublic, "{address}");
        }
    }

    #[test]
    fn an_address_is_public_when_its_literal_or_everything_its_name_resolves_to_is() {
        assert!(resolves_to_public(&Url::parse("https://8.8.8.8/").unwrap()));
        assert!(!resolves_to_public(&Url::parse("http://localhost/").unwrap()));
        assert!(!resolves_to_public(&Url::parse("http://[::1]/").unwrap()));
    }
}
