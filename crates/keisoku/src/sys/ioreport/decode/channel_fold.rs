use super::{FrequencyTables, RawChannel};
use crate::sys::ioreport::IoReportGroups;

pub(crate) trait ChannelFold {
    const GROUPS: IoReportGroups;

    fn wants(channel: &RawChannel) -> bool;

    fn fold(
        &mut self,
        channel: &RawChannel,
        frequencies: Option<&FrequencyTables<'_>>,
    );
}
