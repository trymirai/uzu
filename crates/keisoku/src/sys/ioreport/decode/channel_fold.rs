use super::{Channel, FrequencyTables, RawChannel};

pub(crate) trait ChannelFold {
    fn fold(
        &mut self,
        channel: Channel,
        raw: &RawChannel,
        frequencies: Option<&FrequencyTables<'_>>,
    );
}
