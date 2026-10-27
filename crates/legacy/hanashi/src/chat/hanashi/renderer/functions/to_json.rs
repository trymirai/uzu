use std::io;

/// JSON formatter matching Python's `json.dumps` default separators (`, ` and `: `)
struct SpacedJsonFormatter;

impl serde_json::ser::Formatter for SpacedJsonFormatter {
    fn write_f64<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        value: f64,
    ) -> io::Result<()> {
        // Keep serde_json's correctly rounded digits; adapt only notation to
        // Python's exponent threshold, explicit sign and two-digit minimum.
        let mut bytes = Vec::new();
        serde_json::ser::CompactFormatter.write_f64(&mut bytes, value)?;
        let text = std::str::from_utf8(&bytes).expect("JSON floats are ASCII");
        if let Some((mantissa, exponent)) = text.split_once('e') {
            let exponent: i32 = exponent.parse().expect("float exponent is an integer");
            write!(writer, "{mantissa}e{exponent:+03}")
        } else {
            let unsigned = text.strip_prefix('-').unwrap_or(text);
            if unsigned.starts_with("0.0000") {
                let fraction = &unsigned[2..];
                let digits = fraction.trim_start_matches('0');
                let exponent = fraction.len() - digits.len() + 1;
                let sign = if value.is_sign_negative() {
                    "-"
                } else {
                    ""
                };
                write!(writer, "{sign}{}", &digits[..1])?;
                if digits.len() > 1 {
                    write!(writer, ".{}", &digits[1..])?;
                }
                write!(writer, "e-{exponent:02}")
            } else {
                writer.write_all(&bytes)
            }
        }
    }

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

/// Serialize model prompt JSON with Python-style separators and without HTML escaping.
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
    .map(minijinja::Value::from_safe_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_json_matches_python_float_notation() {
        let mut environment = minijinja::Environment::new();
        environment.add_filter("tojson", to_json);
        let value: serde_json::Value = serde_json::from_str(
            r#"{"z": [1e-7, 1e-5, 0.0001, -0.0, 1e15, 1e16, 1e100, -1.739269179332728e61, -2065594985630696.2], "a": "é<>&"}"#,
        )
        .unwrap();
        let rendered = environment
            .template_from_str("{{ values | tojson }}")
            .unwrap()
            .render(minijinja::context!(values => minijinja::value::Serde(&value)))
            .unwrap();
        assert_eq!(
            rendered,
            "{\"z\": [1e-07, 1e-05, 0.0001, -0.0, 1000000000000000.0, 1e+16, 1e+100, -1.739269179332728e+61, -2065594985630696.2], \"a\": \"é<>&\"}"
        );
    }
}
