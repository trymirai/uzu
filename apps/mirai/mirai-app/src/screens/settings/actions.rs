use gpui::Context;

use super::{setting_kind::SettingKind, view::SettingsView};
use crate::{
    data_ops, engine, native_dialog, settings_state, startup,
    toast::{self, ToastKind},
};

impl SettingsView {
    pub(super) fn export_chats(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        let Some(data) = data_ops::export_chats_zip() else {
            toast::push(cx, "No chats to export", ToastKind::Info);
            return;
        };
        let default_name = data_ops::export_zip_default_name();
        let Some(path) = native_dialog::save_file("Export chats", &default_name, "zip", "ZIP archive") else {
            return;
        };
        if native_dialog::write_bytes(&path, &data) {
            toast::push(cx, "Chats exported", ToastKind::Info);
        } else {
            toast::push(cx, "Export failed", ToastKind::Error);
        }
    }

    pub(super) fn export_logs(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        let Some(data) = data_ops::read_log_bytes() else {
            toast::push(cx, "No log file found", ToastKind::Info);
            return;
        };
        let Some(path) = native_dialog::save_file("Export logs", "mirai.log", "log", "Log file") else {
            return;
        };
        if native_dialog::write_bytes(&path, &data) {
            toast::push(cx, "Logs exported", ToastKind::Info);
        } else {
            toast::push(cx, "Export failed", ToastKind::Error);
        }
    }

    pub(super) fn flip(
        &mut self,
        kind: SettingKind,
        cx: &mut Context<Self>,
    ) {
        let mut settings = settings_state::current(cx);
        match kind {
            SettingKind::Reasoning => settings.reasoning = !settings.reasoning,
            SettingKind::RunOnStartup => {
                settings.run_on_startup = !settings.run_on_startup;
                startup::set(settings.run_on_startup);
            },
            SettingKind::ShowInMenuBar => settings.show_in_menu_bar = !settings.show_in_menu_bar,
            SettingKind::AutoEject => settings.auto_eject = !settings.auto_eject,
            SettingKind::ShareUsage => {
                settings.share_usage_data = !settings.share_usage_data;
                if let Some(engine) = engine::try_engine(cx) {
                    engine.set_usage_reporting(settings.share_usage_data);
                }
            },
        }
        settings_state::set(cx, settings);
    }

    pub(super) fn bump_idle(
        &mut self,
        delta: i32,
        cx: &mut Context<Self>,
    ) {
        let mut settings = settings_state::current(cx);
        let next = settings.idle_timeout_minutes as i32 + delta;
        settings.idle_timeout_minutes = next.clamp(1, 120) as u32;
        settings_state::set(cx, settings);
    }
}
