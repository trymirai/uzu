// Paces streamed text at the rate it arrives, measured over the frames so far.

const WARMUP_MS = 200;
const MAX_CHARS_PER_SEC = 600;
const MIN_ELAPSED_MS = 50;

export class RevealPacer {
  private received: string;
  private readonly baseLength: number;
  private revealed: number;
  private elapsedMs = 0;
  private done = false;

  constructor(baseText: string) {
    this.received = baseText;
    this.baseLength = baseText.length;
    this.revealed = baseText.length;
  }

  append(delta: string): void {
    this.received += delta;
  }

  markDone(): void {
    this.done = true;
  }

  get isCaughtUp(): boolean {
    return this.revealed >= this.received.length;
  }

  reset(): void {
    this.received = "";
    this.revealed = 0;
  }

  // Returns newly visible text, or null if nothing changed this frame.
  step(dtMs: number): string | null {
    if (this.revealed >= this.received.length) return null;
    this.elapsedMs += dtMs;
    if (!this.done && this.elapsedMs < WARMUP_MS) return null;

    const receivedSinceBase = this.received.length - this.baseLength;
    const ratePerMs = Math.min(receivedSinceBase / Math.max(this.elapsedMs, MIN_ELAPSED_MS), MAX_CHARS_PER_SEC / 1000);
    const before = Math.floor(this.revealed);
    this.revealed = Math.min(this.received.length, this.revealed + ratePerMs * dtMs);
    const after = Math.floor(this.revealed);
    return after > before ? this.received.slice(0, after) : null;
  }
}
