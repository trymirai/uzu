use crate::{
    marker::IntervalSet,
    sys::ioreport::{IoReport, IoReportGroups, RawIOReportSample},
};

pub(crate) struct IntervalEngine {
    ioreport: Option<IoReport>,
}

pub(crate) struct IntervalSession {
    begin: Option<RawIOReportSample>,
}

pub(crate) struct IntervalReading {
    begin: Option<RawIOReportSample>,
    end: Option<RawIOReportSample>,
}

impl IntervalEngine {
    pub(crate) fn new(groups: IoReportGroups) -> Self {
        Self {
            ioreport: (!groups.is_empty()).then(|| IoReport::for_groups(groups)).flatten(),
        }
    }

    pub(crate) fn begin(&self) -> IntervalSession {
        IntervalSession {
            begin: self.ioreport.as_ref().and_then(IoReport::snapshot),
        }
    }

    pub(crate) fn end(
        &self,
        session: IntervalSession,
    ) -> IntervalReading {
        IntervalReading {
            begin: session.begin,
            end: self.ioreport.as_ref().and_then(IoReport::snapshot),
        }
    }

    pub(crate) fn fold<M: IntervalSet>(
        &self,
        reading: &IntervalReading,
        values: &mut M::Value,
    ) {
        if let (Some(ioreport), Some(begin), Some(end)) =
            (self.ioreport.as_ref(), reading.begin.as_ref(), reading.end.as_ref())
        {
            ioreport.for_each_channel(begin, end, |channel, raw| {
                M::apply(channel, raw, values);
            });
        }
    }
}
