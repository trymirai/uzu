//! Capture-point macros for activation tracing.
//!
//! These live outside the `trace` module so that call sites compile unchanged
//! whether or not the `trace` feature is enabled: with the feature off every
//! macro expands to nothing, with it on they expand to a method call on the
//! encoder that returns immediately unless a recorder is attached.

/// Records `$src` under `$name` in the encoder's current trace scope.
///
/// ```ignore
/// trace!(encoder, "pre_mixer_norm", &hidden, [1, rows, model_dim], data_type);
/// ```
macro_rules! trace {
    ($encoder:expr, $name:expr, $src:expr, $shape:expr, $data_type:expr $(,)?) => {{
        #[cfg(feature = "trace")]
        $encoder.trace($name, $src, &$shape[..], $data_type);
    }};
}

/// Records host-side `$data` under `$name`, without a device round-trip.
#[cfg_attr(not(feature = "trace"), allow(unused_macros))]
macro_rules! trace_host {
    ($encoder:expr, $name:expr, $data:expr, $shape:expr, $data_type:expr $(,)?) => {{
        #[cfg(feature = "trace")]
        $encoder.trace_host($name, $data, &$shape[..], $data_type);
    }};
}

/// Binds a value that only trace capture points read, so it disappears
/// together with them when the feature is off.
macro_rules! trace_let {
    ($name:ident = $value:expr $(,)?) => {
        #[cfg(feature = "trace")]
        let $name = $value;
    };
}

/// Pushes a path segment; every later capture nests under it until
/// [`trace_scope_end!`]. Takes `format!`-style arguments.
///
/// ```ignore
/// trace_scope!(encoder, "layer_results.{}", layer_index);
/// ```
macro_rules! trace_scope {
    ($encoder:expr, $($segment:tt)+) => {{
        #[cfg(feature = "trace")]
        $encoder.push_trace_scope(format_args!($($segment)+));
    }};
}

/// Pops the segment pushed by the matching [`trace_scope!`].
macro_rules! trace_scope_end {
    ($encoder:expr $(,)?) => {{
        #[cfg(feature = "trace")]
        $encoder.pop_trace_scope();
    }};
}

// `trace_host` only has call sites behind the `trace` feature, unlike the
// others which sit in the shared encode path.
#[cfg_attr(not(feature = "trace"), allow(unused_imports))]
pub(crate) use {trace, trace_host, trace_let, trace_scope, trace_scope_end};
