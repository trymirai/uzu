use std::fmt;

use chrono::Local;
use tracing::{
    Event, Level, Subscriber,
    field::{Field, Visit},
    span::{Attributes, Id},
};
use tracing_subscriber::{
    Layer,
    fmt::{FmtContext, FormatEvent, FormatFields, format::Writer},
    layer::Context,
    registry::LookupSpan,
};

const ANSI_RESET: &str = "\x1b[0m";
const ANSI_DIM: &str = "\x1b[2m";

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

fn level_ansi(level: &Level) -> &'static str {
    match *level {
        Level::ERROR => "\x1b[31m",
        Level::WARN => "\x1b[33m",
        Level::INFO => "\x1b[32m",
        Level::DEBUG => "\x1b[34m",
        Level::TRACE => "\x1b[35m",
    }
}

/// Console/file event format: `<timestamp> <LEVEL> [<request_id>] <fields>`.
///
/// Colors and dimming are emitted only when constructed with
/// [`RequestFormatter::colored`]; the file log uses the plain variant.
pub(super) struct RequestFormatter {
    ansi: bool,
}

impl RequestFormatter {
    pub(super) fn colored() -> Self {
        Self {
            ansi: true,
        }
    }

    pub(super) fn plain() -> Self {
        Self {
            ansi: false,
        }
    }

    fn timestamp(&self) -> String {
        let now = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
        if self.ansi {
            format!("{}{now}{}", ANSI_DIM, ANSI_RESET)
        } else {
            now.to_string()
        }
    }
}

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
        let metadata = event.metadata();
        write!(writer, "{} ", self.timestamp())?;
        let level = metadata.level().to_string();
        let padded = format!("{level:<5}");
        if self.ansi {
            write!(writer, "{}{padded}{}", level_ansi(metadata.level()), ANSI_RESET)?;
        } else {
            write!(writer, "{padded}")?;
        }

        if let Some(scope) = ctx.event_scope() {
            for span in scope {
                let extensions = span.extensions();
                if let Some(RequestId(Some(id))) = extensions.get::<RequestId>() {
                    if self.ansi {
                        write!(writer, " \x1b[36m[{id}]\x1b[0m")?;
                    } else {
                        write!(writer, " [{id}]")?;
                    }
                    break;
                }
            }
        }

        write!(writer, " ")?;
        ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}
