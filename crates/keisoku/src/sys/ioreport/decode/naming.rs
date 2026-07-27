use obfstr::obfstr;

use crate::sys::ioreport::kinds::DramFlow;

pub(crate) fn strip_die_prefix(channel: &str) -> &str {
    let Some(rest) = channel.strip_prefix(obfstr!("DIE")) else {
        return channel;
    };
    let rest = rest.trim_start_matches(|character: char| character.is_ascii_digit());
    rest.strip_prefix(' ').unwrap_or(channel)
}

pub(crate) fn dcs_flow(aggregate: &str) -> Option<DramFlow> {
    if aggregate == obfstr!("DCS RD") {
        Some(DramFlow::DramRead)
    } else if aggregate == obfstr!("DCS WR") {
        Some(DramFlow::DramWrite)
    } else {
        None
    }
}

pub(crate) fn read_write_flow(name: &str) -> Option<DramFlow> {
    if name.ends_with(obfstr!(" RD")) {
        Some(DramFlow::DramRead)
    } else if name.ends_with(obfstr!(" WR")) {
        Some(DramFlow::DramWrite)
    } else {
        None
    }
}
