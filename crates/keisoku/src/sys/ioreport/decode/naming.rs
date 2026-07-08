use obfstr::obfstr;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DramFlow {
    Read,
    Write,
}

pub(crate) fn strip_die_prefix(channel: &str) -> &str {
    let Some(rest) = channel.strip_prefix(obfstr!("DIE")) else {
        return channel;
    };
    let rest = rest.trim_start_matches(|character: char| character.is_ascii_digit());
    rest.strip_prefix(' ').unwrap_or(channel)
}

pub(crate) fn dcs_flow(aggregate: &str) -> Option<DramFlow> {
    if aggregate == obfstr!("DCS RD") {
        Some(DramFlow::Read)
    } else if aggregate == obfstr!("DCS WR") {
        Some(DramFlow::Write)
    } else {
        None
    }
}
