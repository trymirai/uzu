pub fn strftime_now(format_string: String) -> String {
    chrono::Local::now().format(&format_string).to_string()
}
