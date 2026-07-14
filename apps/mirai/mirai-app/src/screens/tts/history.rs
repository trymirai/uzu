use std::fs;

use gpui::{Context, PathPromptOptions};

use super::view::TtsView;
use crate::tts_history;

impl TtsView {
    pub(super) fn pick_text_file(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        let rx = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Open text file".into()),
        });
        cx.spawn(async move |this, cx| {
            let Ok(Ok(Some(paths))) = rx.await else {
                return;
            };
            let Some(path) = paths.first() else {
                return;
            };
            if let Ok(content) = fs::read_to_string(path) {
                let _ = this.update(cx, |this, cx| {
                    this.input.update(cx, |input, cx| input.set_text(&content, cx));
                    cx.notify();
                });
            }
        })
        .detach();
    }

    pub(super) fn reload_history(&mut self) {
        self.history = tts_history::list();
    }

    pub fn reload_after_clear(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        self.stop_playback(cx);
        self.generation_id = self.generation_id.wrapping_add(1);
        self.reload_history();
        cx.notify();
    }

    pub(super) fn play_history(
        &mut self,
        id: &str,
        cx: &mut Context<Self>,
    ) {
        if self.playing_id.as_deref() == Some(id) {
            self.stop_playback(cx);
            return;
        }
        let Some(batch) = tts_history::load_pcm(id) else {
            self.error = Some("audio file missing".into());
            cx.notify();
            return;
        };
        if let Some(player) = &self.player {
            player.stop();
        }
        if self.append_pcm(batch).is_err() {
            cx.notify();
            return;
        }
        self.playing_id = Some(id.to_string());
        self.error = None;
        cx.notify();
    }

    pub(super) fn stop_playback(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        if let Some(player) = &self.player {
            player.stop();
        }
        self.playing_id = None;
        cx.notify();
    }

    pub(super) fn tick_playback(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        if self.playing_id.is_some() && self.player.as_ref().is_some_and(|p| p.is_finished()) {
            self.playing_id = None;
            cx.notify();
        }
    }

    pub(super) fn delete_history(
        &mut self,
        id: &str,
        cx: &mut Context<Self>,
    ) {
        if self.playing_id.as_deref() == Some(id) {
            self.stop_playback(cx);
        }
        tts_history::delete(id);
        self.reload_history();
        cx.notify();
    }

    pub(super) fn restore_text(
        &mut self,
        text: &str,
        cx: &mut Context<Self>,
    ) {
        self.input.update(cx, |input, cx| input.set_text(text, cx));
        cx.notify();
    }
}
