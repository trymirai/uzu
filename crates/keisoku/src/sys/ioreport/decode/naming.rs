use obfstr::obfstr;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DramFlow {
    Read,
    Write,
    Combined,
}

pub(crate) fn strip_die_prefix(channel: &str) -> &str {
    let Some(rest) = channel.strip_prefix(obfstr!("DIE")) else {
        return channel;
    };
    let rest = rest.trim_start_matches(|character: char| character.is_ascii_digit());
    rest.strip_prefix(' ').unwrap_or(channel)
}

pub(crate) fn dram_flow(channel: &str) -> Option<DramFlow> {
    if channel.contains(obfstr!("RD+WR")) || channel.ends_with(obfstr!("RW")) {
        Some(DramFlow::Combined)
    } else if channel.contains(obfstr!("WR")) {
        Some(DramFlow::Write)
    } else if channel.contains(obfstr!("RD")) {
        Some(DramFlow::Read)
    } else {
        None
    }
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
