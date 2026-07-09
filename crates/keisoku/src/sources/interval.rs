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

    pub(crate) fn fold_end<M: IntervalSet>(
        &self,
        session: IntervalSession,
        values: &mut M::Value,
    ) {
        let Some(ioreport) = self.ioreport.as_ref() else {
            return;
        };
        let Some(begin) = session.begin.as_ref() else {
            return;
        };
        let Some(end) = ioreport.snapshot() else {
            return;
        };
        ioreport.for_each_channel(begin, &end, |channel, raw| {
            M::apply(channel, raw, values);
        });
    }
}
