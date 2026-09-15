use std::fmt;

use tracing::{
    Event, Subscriber,
    field::{Field, Visit},
    span::{Attributes, Id},
};
use tracing_subscriber::{
    Layer,
    fmt::{FmtContext, FormatEvent, FormatFields, format::Writer},
    layer::Context,
    registry::LookupSpan,
};

#[derive(Default)]
struct RequestId(Option<String>);

impl Visit for RequestId {
    fn record_str(
        &mut self,
        field: &Field,
        value: &str,
    ) {
        if field.name() == "request_id" {
            self.0 = Some(value.to_owned());
        }
    }

    fn record_debug(
        &mut self,
        field: &Field,
        value: &dyn fmt::Debug,
    ) {
        if field.name() == "request_id" {
            self.0 = Some(format!("{value:?}"));
        }
    }
}

/// Store the request ID separately from the span's human-readable fields.
pub(super) struct RequestContextLayer;

impl<S> Layer<S> for RequestContextLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &Attributes<'_>,
        id: &Id,
        ctx: Context<'_, S>,
    ) {
        let mut request_id = RequestId::default();
        attrs.record(&mut request_id);
        if request_id.0.is_some()
            && let Some(span) = ctx.span(id)
        {
            span.extensions_mut().insert(request_id);
        }
    }
}

pub(super) struct RequestFormatter;

impl<S, N> FormatEvent<S, N> for RequestFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        if let Some(scope) = ctx.event_scope() {
            for span in scope {
                let extensions = span.extensions();
                if let Some(RequestId(Some(id))) = extensions.get::<RequestId>() {
                    write!(writer, "[{id}] ")?;
                    break;
                }
            }
        }
        ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}
