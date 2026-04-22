use std::{fs, path::PathBuf};

use regex::Regex;

use crate::types::{Error, Snippet};

pub fn extract_snippets(path: PathBuf) -> Result<Vec<Snippet>, Error> {
    let start_re = Regex::new(r"(?m)^\s*//\s*snippet:([^\s]+)\s*$").map_err(|_| Error::UnableToExtractSnippets)?;
    let mut results = Vec::new();

    let files = fs::read_dir(path).map_err(|_| Error::UnableToExtractSnippets)?;
    for file in files {
        let file = file.map_err(|_| Error::UnableToExtractSnippets)?;
        let path = file.path();
        let contents = fs::read_to_string(path.clone()).map_err(|_| Error::UnableToExtractSnippets)?;

        for start_cap in start_re.captures_iter(&contents) {
            let name = start_cap.get(1).ok_or_else(|| Error::UnableToExtractSnippets)?.as_str().to_string();
            let body_start = start_cap.get(0).ok_or_else(|| Error::UnableToExtractSnippets)?.end();

            let end_re = Regex::new(&format!(r"(?m)^\s*//\s*endsnippet:{}\s*$", regex::escape(&name)))
                .map_err(|_| Error::UnableToExtractSnippets)?;

            let end_match = end_re.find_at(&contents, body_start).ok_or_else(|| Error::UnableToExtractSnippets)?;

            let mut content = &contents[body_start..end_match.start()];
            if let Some(stripped) = content.strip_prefix('\n') {
                content = stripped;
            }
            let content = content.trim_end().to_string();
            let content = align_to_first_line_indent(&content).to_string();

            let snippet = Snippet::new(name, content, path.clone());
            results.push(snippet);
        }
    }

    Ok(results)
}

fn align_to_first_line_indent(body: &str) -> String {
    // Trim leading/trailing blank lines first
    let mut lines: Vec<&str> = body.lines().collect();

    // leading blanks
    while lines.first().is_some_and(|l| l.trim().is_empty()) {
        lines.remove(0);
    }
    // trailing blanks
    while lines.last().is_some_and(|l| l.trim().is_empty()) {
        lines.pop();
    }

    if lines.is_empty() {
        return String::new();
    }

    // Determine the exact whitespace prefix (spaces/tabs) of the first non-empty line
    let first_non_empty_idx = lines.iter().position(|l| !l.trim().is_empty()).unwrap_or(0);
    let first = lines[first_non_empty_idx];
    let indent_prefix: String = first.chars().take_while(|c| *c == ' ' || *c == '\t').collect();

    // Strip exactly that prefix from every line when present; leave others untouched
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        let dedented = line.strip_prefix(&indent_prefix).unwrap_or(line);
        if i > 0 {
            out.push('\n');
        }
        out.push_str(dedented);
    }
    out
}
