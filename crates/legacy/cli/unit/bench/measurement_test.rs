use std::io::Cursor;

use super::*;

#[test]
fn measurement_requires_start_and_stop_acknowledgements() {
    let mut input = Cursor::new(b"run\nack\n");
    let mut output = Vec::new();
    exchange(&mut input, &mut output, "ready", "run").unwrap();
    exchange(&mut input, &mut output, "done", "ack").unwrap();
    assert_eq!(output, b"\n{\"benchmark_event\":\"ready\"}\n\n{\"benchmark_event\":\"done\"}\n");
}

#[test]
fn measurement_fails_when_controller_disconnects_or_sends_wrong_command() {
    for reply in ["", "run", "run \n", "ack\n", "run extra\n"] {
        assert!(exchange(&mut Cursor::new(reply), &mut Vec::new(), "ready", "run").is_err());
    }
}
