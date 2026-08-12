// Defined outside the cfg-gated `trace` module so call sites compile either way.

macro_rules! trace {
    ($encoder:expr, $name:expr, $src:expr, $shape:expr, $data_type:expr $(,)?) => {{
        #[cfg(feature = "trace")]
        $encoder.trace($name, $src, &$shape[..], $data_type);
    }};
}

#[cfg_attr(not(feature = "trace"), allow(unused_macros))]
macro_rules! trace_host {
    ($encoder:expr, $name:expr, $data:expr, $shape:expr, $data_type:expr $(,)?) => {{
        #[cfg(feature = "trace")]
        $encoder.trace_host($name, $data, &$shape[..], $data_type);
    }};
}

macro_rules! trace_scope {
    ($encoder:expr, $($segment:tt)+) => {{
        #[cfg(feature = "trace")]
        $encoder.push_trace_scope(format_args!($($segment)+));
    }};
}

macro_rules! trace_scope_end {
    ($encoder:expr $(,)?) => {{
        #[cfg(feature = "trace")]
        $encoder.pop_trace_scope();
    }};
}

#[cfg_attr(not(feature = "trace"), allow(unused_imports))]
pub(crate) use {trace, trace_host, trace_scope, trace_scope_end};
