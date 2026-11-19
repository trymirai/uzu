// Release builds are GUI executables, so Windows opens no console beside the window.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    mirai_gui::run()
}
