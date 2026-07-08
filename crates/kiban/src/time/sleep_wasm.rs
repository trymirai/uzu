#![cfg(target_family = "wasm")]

use std::{
    cell::RefCell,
    future::Future,
    pin::Pin,
    rc::Rc,
    task::{Context, Poll, Waker},
    time::Duration,
};

use wasm_bindgen::{JsCast, closure::Closure};

struct SleepState {
    done: bool,
    waker: Option<Waker>,
}

pub(crate) struct Sleep {
    state: Rc<RefCell<SleepState>>,
    _done: Closure<dyn FnMut()>,
}

impl Sleep {
    pub fn new(duration: Duration) -> Self {
        let state = Rc::new(RefCell::new(SleepState {
            done: false,
            waker: None,
        }));

        let state_clone = state.clone();
        let done = Closure::wrap(Box::new(move || {
            let mut s = state_clone.borrow_mut();
            s.done = true;
            if let Some(waker) = s.waker.take() {
                waker.wake();
            }
        }) as Box<dyn FnMut()>);

        let window = web_sys::window().expect("no global `window` exists");
        window
            .set_timeout_with_callback_and_timeout_and_arguments_0(
                done.as_ref().unchecked_ref(),
                duration.as_millis() as i32,
            )
            .expect("failed to set timeout");

        Sleep {
            state,
            _done: done,
        }
    }
}

impl Future for Sleep {
    type Output = ();

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<()> {
        let mut state = self.state.borrow_mut();
        if state.done {
            Poll::Ready(())
        } else {
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

pub fn sleep(duration: Duration) -> Sleep {
    Sleep::new(duration)
}
