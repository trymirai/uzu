const WARMUP_SECONDS: f32 = 0.2;
const MAX_REVEAL_PER_SECOND: f32 = 600.0;
const MIN_ELAPSED_SECONDS: f32 = 0.05;

pub(super) struct RevealPacer {
    revealed: f32,
    elapsed: f32,
    show_all: bool,
}

impl Default for RevealPacer {
    fn default() -> Self {
        Self::shown()
    }
}

impl RevealPacer {
    pub(super) fn shown() -> Self {
        Self {
            revealed: 0.0,
            elapsed: 0.0,
            show_all: true,
        }
    }

    pub(super) fn streaming() -> Self {
        Self {
            revealed: 0.0,
            elapsed: 0.0,
            show_all: false,
        }
    }

    pub(super) fn revealed_chars(&self) -> usize {
        if self.show_all {
            usize::MAX
        } else {
            self.revealed as usize
        }
    }

    pub(super) fn advance(
        &mut self,
        received: usize,
        dt: f32,
        done: bool,
    ) -> bool {
        if self.show_all {
            return false;
        }
        if received > 0 {
            self.elapsed += dt;
        }
        if !done && self.elapsed < WARMUP_SECONDS {
            return true;
        }
        let received = received as f32;
        let rate = (received / self.elapsed.max(MIN_ELAPSED_SECONDS)).min(MAX_REVEAL_PER_SECOND);
        self.revealed = (self.revealed + rate * dt).min(received);
        !done || self.revealed < received
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 0.016;
    const MAX_STEP: usize = (MAX_REVEAL_PER_SECOND * DT) as usize + 1;

    struct Trace {
        revealed: Vec<usize>,
        received: Vec<usize>,
    }

    fn run(
        received_at_frame: impl Fn(usize) -> usize,
        frames: usize,
        done_frame: usize,
    ) -> Trace {
        let mut pacer = RevealPacer::streaming();
        let mut revealed = Vec::with_capacity(frames);
        let mut received = Vec::with_capacity(frames);
        for frame in 0..frames {
            let got = received_at_frame(frame);
            pacer.advance(got, DT, frame >= done_frame);
            revealed.push(pacer.revealed_chars());
            received.push(got);
        }
        Trace {
            revealed,
            received,
        }
    }

    fn max_step(revealed: &[usize]) -> usize {
        revealed.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0)
    }

    fn max_zero_run_with_pending(trace: &Trace) -> usize {
        let mut run = 0usize;
        let mut worst = 0usize;
        for frame in 1..trace.revealed.len() {
            let pending = trace.received[frame] > trace.revealed[frame];
            let stalled = trace.revealed[frame] == trace.revealed[frame - 1];
            if pending && stalled {
                run += 1;
                worst = worst.max(run);
            } else {
                run = 0;
            }
        }
        worst
    }

    fn assert_never_exceeds_received(trace: &Trace) {
        for frame in 0..trace.revealed.len() {
            assert!(
                trace.revealed[frame] <= trace.received[frame],
                "frame {frame}: revealed {} > received {}",
                trace.revealed[frame],
                trace.received[frame]
            );
        }
    }

    fn assert_monotonic(revealed: &[usize]) {
        for frame in 1..revealed.len() {
            assert!(revealed[frame] >= revealed[frame - 1], "reveal went backwards at frame {frame}");
        }
    }

    fn coefficient_of_variation(samples: &[f32]) -> f32 {
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        if mean <= 0.0 {
            return 0.0;
        }
        let variance = samples.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / samples.len() as f32;
        variance.sqrt() / mean
    }

    #[test]
    fn bursty_arrival_never_stalls() {
        // 60 chars arrive every 10 frames (~160ms gaps); the standing buffer must bridge the gaps.
        let trace = run(|frame| (frame / 10 + 1) * 60, 200, 200);
        assert_monotonic(&trace.revealed);
        assert_never_exceeds_received(&trace);
        let warm = Trace {
            revealed: trace.revealed[30..].to_vec(),
            received: trace.received[30..].to_vec(),
        };
        assert!(max_zero_run_with_pending(&warm) <= 2, "stall detected: {}", max_zero_run_with_pending(&warm));
        assert!(max_step(&trace.revealed) <= MAX_STEP, "chunky jump: {}", max_step(&trace.revealed));
    }

    #[test]
    fn constant_speed_under_bursty_arrival() {
        // The key property: bursty INPUT must produce a steady OUTPUT rate, not one that lurches up and down.
        let trace = run(|frame| (frame / 10 + 1) * 60, 200, 200);
        let windows: Vec<f32> =
            (20..160).step_by(10).map(|frame| (trace.revealed[frame + 10] - trace.revealed[frame]) as f32).collect();
        let cv = coefficient_of_variation(&windows);
        assert!(cv < 0.2, "reveal speed varies too much across windows (cv={cv:.3})");
    }

    #[test]
    fn steady_arrival_is_smooth() {
        let trace = run(|frame| frame * 5, 200, 200);
        assert_monotonic(&trace.revealed);
        assert_never_exceeds_received(&trace);
        assert!(max_step(&trace.revealed) <= MAX_STEP, "chunky jump: {}", max_step(&trace.revealed));
        // Skip the ~0.2s warmup (reveal is intentionally 0 while the cushion builds; the loader shows).
        let warm = Trace {
            revealed: trace.revealed[30..].to_vec(),
            received: trace.received[30..].to_vec(),
        };
        assert!(max_zero_run_with_pending(&warm) <= 2, "stall: {}", max_zero_run_with_pending(&warm));
    }

    #[test]
    fn completes_after_done_without_dumping() {
        // Whole message present immediately, stream already done: reveal as a smooth typewriter, no dump.
        let trace = run(|_| 1000, 160, 0);
        assert_monotonic(&trace.revealed);
        assert!(max_step(&trace.revealed) <= MAX_STEP, "dumped {} chars in one frame", max_step(&trace.revealed));
        assert_eq!(*trace.revealed.last().unwrap(), 1000, "did not finish revealing");
    }

    #[test]
    fn drains_remaining_buffer_after_generation_stops() {
        // Fast generation for 40 frames, then stops and is marked done; the leftover buffer must drain.
        let trace = run(|frame| frame.min(40) * 20, 160, 40);
        assert_never_exceeds_received(&trace);
        assert_eq!(*trace.revealed.last().unwrap(), 800, "buffer not drained after done");
        assert!(max_step(&trace.revealed) <= MAX_STEP, "chunky drain: {}", max_step(&trace.revealed));
    }

    #[test]
    fn reveal_lags_generation_by_a_bounded_amount() {
        let trace = run(|frame| frame * 6, 300, 300);
        let worst_lag = (30..trace.revealed.len()).map(|f| trace.received[f] - trace.revealed[f]).max().unwrap_or(0);
        assert!(worst_lag <= 200, "reveal lagged generation by {worst_lag} chars");
    }
}
