use std::io;

/// JSON formatter matching Python's `json.dumps` default separators (`, ` and `: `)
struct SpacedJsonFormatter;

impl serde_json::ser::Formatter for SpacedJsonFormatter {
    fn begin_object_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
    ) -> io::Result<()> {
        writer.write_all(b": ")
    }

    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
}

/// Copied from minijinja::filters::tojson with SpacedJsonFormatter
pub fn to_json(
    value: &minijinja::Value,
    indent: Option<minijinja::Value>,
    args: minijinja::value::Kwargs,
) -> Result<minijinja::Value, minijinja::Error> {
    let indent = match indent {
        Some(indent) => Some(indent),
        None => args.get("indent")?,
    };
    let indent = match indent {
        None => None,
        Some(ref val) => match bool::try_from(val.clone()).ok() {
            Some(true) => Some(2),
            Some(false) => None,
            None => Some(usize::try_from(val.clone())?),
        },
    };
    args.assert_all_used()?;
    if let Some(indent) = indent {
        let mut out = Vec::<u8>::new();
        let indentation = " ".repeat(indent);
        let formatter = serde_json::ser::PrettyFormatter::with_indent(indentation.as_bytes());
        let mut s = serde_json::Serializer::with_formatter(&mut out, formatter);
        serde::Serialize::serialize(&value, &mut s).map(|_| unsafe { String::from_utf8_unchecked(out) })
    } else {
        let mut out = Vec::<u8>::new();
        let mut s = serde_json::Serializer::with_formatter(&mut out, SpacedJsonFormatter);
        serde::Serialize::serialize(&value, &mut s).map(|_| unsafe { String::from_utf8_unchecked(out) })
    }
    .map_err(|err| {
        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "cannot serialize to JSON").with_source(err)
    })
    .map(|s| {
        // When this filter is used the return value is safe for both HTML and JSON
        let mut rv = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '<' => rv.push_str("\\u003c"),
                '>' => rv.push_str("\\u003e"),
                '&' => rv.push_str("\\u0026"),
                '\'' => rv.push_str("\\u0027"),
                _ => rv.push(c),
            }
        }
        minijinja::Value::from_safe_string(rv)
    })
}
