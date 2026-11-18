pub fn strftime_now(format_string: String) -> String {
    jiff::Zoned::now().strftime(&format_string).to_string()
}
