mod report;

use std::{
    io::{BufRead, BufReader, Write},
    process::{Command, Stdio},
    time::Duration,
};

use keisoku::{Config, start};

const MARK_PREFIX: &str = "KEISOKU_MARK ";

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("keisoku") {
        args.remove(0);
    }

    let mut interval_ms = 100u64;
    let mut out_path = String::from("keisoku-report.json");
    while let Some(flag) = args.first().cloned() {
        match flag.as_str() {
            "--interval-ms" => {
                args.remove(0);
                interval_ms = take_value(&mut args, "--interval-ms").parse().expect("invalid --interval-ms value");
            },
            "--out" => {
                args.remove(0);
                out_path = take_value(&mut args, "--out");
            },
            _ => break,
        }
    }

    if args.is_empty() {
        eprintln!("usage: cargo keisoku [--interval-ms N] [--out PATH] <cargo subcommand...>");
        std::process::exit(2);
    }

    let recorder = start(Config {
        interval: Duration::from_millis(interval_ms),
    });

    let mut child = Command::new("cargo").args(&args).stdout(Stdio::piped()).spawn().expect("failed to spawn cargo");

    let stdout = child.stdout.take().expect("missing child stdout");
    let mut forward = std::io::stdout();
    for line in BufReader::new(stdout).lines() {
        let line = line.unwrap_or_default();
        if let Some(label) = line.strip_prefix(MARK_PREFIX) {
            recorder.mark(label.trim());
        }
        let _ = writeln!(forward, "{line}");
    }

    let status = child.wait().expect("failed to wait for cargo");
    let report = report::build(recorder.stop());
    let json = serde_json::to_string_pretty(&report).expect("failed to serialize report");
    std::fs::write(&out_path, json).expect("failed to write report");
    eprintln!("keisoku: wrote {out_path} ({} windows)", report.windows.len());

    std::process::exit(status.code().unwrap_or(1));
}

fn take_value(
    args: &mut Vec<String>,
    flag: &str,
) -> String {
    if args.is_empty() {
        eprintln!("{flag} requires a value");
        std::process::exit(2);
    }
    args.remove(0)
}
