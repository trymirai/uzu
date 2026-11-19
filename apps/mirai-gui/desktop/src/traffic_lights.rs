//! The window's traffic lights. They sit level with the sidebar button beside them, and follow it
//! as the interface zooms: the button grows from a fixed top edge (see the left bar), so its centre
//! drops, and the lights step down with it, as in Claude and Codex.
//!
//! tao keeps the lights where the window was created with them, on every redraw, so a resize never
//! moves them. It sets their left edge and the height of the title bar that holds them, never their
//! height within it. So the window is created with the title bar deep enough for the lowest place
//! the lights take (at the largest zoom), and zooming only moves them up and down inside it. AppKit
//! sometimes resets that height on a resize; the buttons are watched and lifted again at once.

use std::{
    ptr::NonNull,
    sync::{
        OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use block2::RcBlock;
use objc2::{MainThreadMarker, rc::Retained};
use objc2_app_kit::{NSButton, NSView, NSViewFrameDidChangeNotification, NSWindow, NSWindowButton, NSWindowStyleMask};
use objc2_foundation::{NSNotification, NSNotificationCenter};
use tauri::{LogicalPosition, Manager, WebviewWindow};

/// Where the lights sit at 100%: the close button's left edge, and its distance from the top.
const X: f64 = 20.0;
const Y: f64 = 28.0;
/// How far the lights drop per unit of zoom: half the sidebar button's 32px height, which is how
/// far its centre drops when it grows.
const DROP_PER_ZOOM: f64 = 16.0;
/// The largest zoom (desktop/src/zoom.rs), where the lights sit lowest.
const MAX_ZOOM: f64 = 1.75;
const MAX_DROP: f64 = DROP_PER_ZOOM * (MAX_ZOOM - 1.0);

/// Where tao keeps the lights: the lowest place they take. `place` lifts them from there.
pub fn creation_position() -> LogicalPosition<f64> {
    LogicalPosition::new(X, Y + MAX_DROP)
}

/// How far above their lowest place the lights sit at a zoom level.
fn lift_for(zoom: f64) -> f64 {
    MAX_DROP - DROP_PER_ZOOM * (zoom - 1.0)
}

/// The buttons' height in their title bar as AppKit lays them out, before any lift.
static RESTING_Y: OnceLock<f64> = OnceLock::new();
/// The lift for the current zoom (f64 bits), read again whenever AppKit moves the buttons.
static LIFT: AtomicU64 = AtomicU64::new(0);
static WATCHING: AtomicBool = AtomicBool::new(false);

/// Moves the lights to their height for the window's current zoom. Must run on the main thread;
/// from anywhere else it hands itself over to it.
pub fn place(window: &WebviewWindow) {
    if MainThreadMarker::new().is_none() {
        let target = window.clone();
        let _ = window.run_on_main_thread(move || place(&target));
        return;
    }
    let zoom = crate::zoom::current(window.app_handle());
    LIFT.store(lift_for(zoom).to_bits(), Ordering::Relaxed);
    let Ok(pointer) = window.ns_window() else {
        return;
    };
    // SAFETY: tauri hands out the window's live NSWindow, and we're on the main thread, where AppKit
    // expects to be called.
    let ns_window = unsafe { &*(pointer as *const NSWindow) };
    apply(ns_window);
    if !WATCHING.swap(true, Ordering::Relaxed) {
        watch(ns_window, pointer as usize);
    }
}

fn buttons(window: &NSWindow) -> [Option<Retained<NSButton>>; 3] {
    [NSWindowButton::CloseButton, NSWindowButton::MiniaturizeButton, NSWindowButton::ZoomButton]
        .map(|kind| window.standardWindowButton(kind))
}

fn apply(window: &NSWindow) {
    // In full screen the lights live in macOS's own title bar, which slides down from the top: they
    // stay where macOS puts them there. Leaving full screen lays them out again, and they're lifted
    // back.
    if window.styleMask().contains(NSWindowStyleMask::FullScreen) {
        return;
    }
    let buttons = buttons(window);
    let Some(close) = &buttons[0] else {
        return;
    };
    let lift = f64::from_bits(LIFT.load(Ordering::Relaxed));
    // SAFETY: a plain view-hierarchy read on the main thread.
    let flipped = unsafe { close.superview() }.is_some_and(|title_bar| title_bar.isFlipped());
    let resting = *RESTING_Y.get_or_init(|| NSView::frame(close).origin.y);
    let y = if flipped {
        resting - lift
    } else {
        resting + lift
    };
    for button in buttons.iter().flatten() {
        let mut origin = NSView::frame(button).origin;
        if origin.y != y {
            origin.y = y;
            button.setFrameOrigin(origin);
        }
    }
}

/// AppKit lays the title bar out again on some resizes (a change of width, full screen) and puts
/// the buttons back at their resting height. Their frame-change notice arrives as it happens, before
/// the window draws, so lifting them again there never shows a jump.
fn watch(
    window: &NSWindow,
    pointer: usize,
) {
    let center = NSNotificationCenter::defaultCenter();
    let block = RcBlock::new(move |_: NonNull<NSNotification>| {
        // SAFETY: the window lives as long as the app (closing only hides it), and frame-change
        // notices for its buttons are posted on the main thread.
        apply(unsafe { &*(pointer as *const NSWindow) });
    });
    for button in buttons(window).iter().flatten() {
        button.setPostsFrameChangedNotifications(true);
        // SAFETY: observing a view's own frame-change notice, with no queue, so the block runs
        // synchronously on the thread that moved it (the main thread).
        let token = unsafe {
            center.addObserverForName_object_queue_usingBlock(
                Some(NSViewFrameDidChangeNotification),
                Some(button),
                None,
                &block,
            )
        };
        // Watched for the life of the app.
        std::mem::forget(token);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lights_drop_as_the_interface_grows() {
        assert_eq!(lift_for(MAX_ZOOM), 0.0);
        assert_eq!(lift_for(1.0), MAX_DROP);
        assert!(lift_for(0.75) > lift_for(1.0));
    }
}
