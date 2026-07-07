use obfstr::obfstr;

pub(crate) fn strip_die_prefix(channel: &str) -> &str {
    let Some(rest) = channel.strip_prefix(obfstr!("DIE")) else {
        return channel;
    };
    let rest = rest.trim_start_matches(|character: char| character.is_ascii_digit());
    rest.strip_prefix(' ').unwrap_or(channel)
}

pub(crate) fn dram_flow(channel: &str) -> Option<bool> {
    if channel.contains(obfstr!("RD+WR")) || channel.ends_with(obfstr!("RW")) {
        None
    } else if channel.contains(obfstr!("WR")) {
        Some(false)
    } else if channel.contains(obfstr!("RD")) {
        Some(true)
    } else {
        None
    }
}
