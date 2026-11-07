import { type RefObject, useEffect, useRef, useState } from "react";
import { useMediaQuery } from "@/hooks/use-media-query";

/*
 * The behaviour engine for the matrix face. It moves like a face, not like a UI:
 * - Gaze moves in saccades: a fast jump, then a fixation that holds.
 * - Blinks are lids, not a squash: the top lid travels most of the way down, the bottom one a
 *   little up. ~75 ms to close, a 25 ms hold, ~150 ms to open with a hair of overshoot. The right
 *   eye trails by 12 ms, a quarter of blinks come in pairs, and blinks like to ride on a glance.
 * - Expressions ease, so one state melts into the next.
 *
 * The Story is uzu's day as an inference engine: what it is doing, as a feeling.
 */

export type StateId =
  | "resting"
  | "waking"
  | "loading"
  | "warming"
  | "reading"
  | "thinking"
  | "drafting"
  | "writing"
  | "tool"
  | "memory"
  | "freeing"
  | "done"
  | "local"
  | "idle"
  | "follow"
  | "ready"
  | "welcome"
  | "watch"
  | "grumpy";

export const STATE_LABEL: Record<StateId, string> = {
  resting: "Resting",
  waking: "Waking up",
  loading: "Loading model",
  warming: "Warming up",
  reading: "Reading prompt",
  thinking: "Thinking",
  drafting: "Drafting tokens",
  writing: "Writing answer",
  tool: "Calling a tool",
  memory: "Memory tight",
  freeing: "Freeing memory",
  done: "Done",
  local: "Stays on device",
  idle: "Idle",
  follow: "Watching you",
  ready: "Ready for you",
  welcome: "Introducing uzu by mirai labs",
  watch: "Watching you type",
  grumpy: "Not pleased",
};

/** Tab names where they differ from the caption. */
export const TAB_LABEL: Partial<Record<StateId, string>> = { welcome: "Welcome" };

/** About forty seconds: a model wakes, loads, reads, thinks, answers, and goes back to rest. */
export const STORY: ReadonlyArray<[StateId, number]> = [
  ["resting", 3.2],
  ["waking", 3.4],
  ["loading", 3.6],
  ["warming", 2.2],
  ["reading", 3.8],
  ["thinking", 2.8],
  ["drafting", 2.6],
  ["writing", 3.6],
  ["tool", 2.4],
  ["memory", 2.2],
  ["freeing", 2.2],
  ["writing", 2.4],
  ["done", 2.6],
  ["local", 7.6],
  ["resting", 3.6],
];
const STORY_LENGTH = STORY.reduce((sum, [, d]) => sum + d, 0);

export type Look = "eyes" | "text";

export type Face = {
  gx: number;
  gy: number;
  blinkL: number; // 0 open .. 1 closed
  blinkR: number;
  squint: number;
  happy: number;
  size: number; // eye scale
  fill: number; // < 0: solid eyes; 0..1: outlined eyes filling from the bottom
  turn: number; // -1..1, a hint of head turn: the near eye widens
  lift: number; // vertical offset of the eyes, in box units
  bracketL: number; // extra outward offset per bracket, in box units
  bracketR: number;
  asym: number; // the right eye narrows while thinking
  sleep: number;
  jitter: number;
  /** What the eyes are drawn as. Switches happen behind a blink. */
  look: Look;
  /** Writing: tokens written so far (carried across visits until the next prompt). */
  text: number;
  /** Whole-face sideways offset, for trembling, in box units. */
  shake: number;
  /** Stays on device: how far the face has turned into an iPhone, and into a Mac (0..1 each). */
  phone: number;
  mac: number;
  /** Head tilt in degrees, around the face's centre. */
  tilt: number;
  /** Whole-face vertical offset (hops, purring), in box units. */
  bob: number;
  /** Reading: square glasses, 0 off .. 1 on. */
  glasses: number;
  /** Cat ears: height scale, outward tilt per ear in degrees (flicks included). */
  earH: number;
  earL: number;
  earR: number;
  caption: string;
  state: StateId;
  t: number;
  zs: ReadonlyArray<{ id: number; age: number }>;
  /** A slow breathing cycle, -1..1. */
  breath: number;
  /** Calling a tool: the eyes' rotation in degrees, as gears. */
  spin: number;
};

export const REST: Face = {
  gx: 0,
  gy: 0,
  blinkL: 0,
  blinkR: 0,
  squint: 0,
  happy: 0,
  size: 1,
  fill: -1,
  turn: 0,
  lift: 0,
  bracketL: 0,
  bracketR: 0,
  asym: 0,
  sleep: 0,
  jitter: 0,
  look: "eyes",
  text: 0,
  shake: 0,
  phone: 0,
  mac: 0,
  tilt: 0,
  bob: 0,
  glasses: 0,
  earH: 1,
  earL: 0,
  earR: 0,
  caption: "",
  state: "idle",
  t: 0,
  zs: [],
  breath: 0,
  spin: 0,
};

export function stateAt(state: string, t: number): { id: StateId; local: number; duration: number } {
  if (state !== "story") return { id: state as StateId, local: t, duration: Infinity };
  let at = t % STORY_LENGTH;
  for (const [id, d] of STORY) {
    if (at < d) return { id, local: at, duration: d };
    at -= d;
  }
  return { id: "idle", local: 0, duration: Infinity };
}

const approach = (from: number, to: number, rate: number, dt: number) =>
  from + (to - from) * (1 - Math.exp(-rate * dt));
const easeIn = (x: number) => x * x * x;
const easeOutBack = (x: number) => {
  const c = 1.2;
  return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2;
};

/** A lid: 100 ms close, 50 ms hold, 220 ms open with a touch of overshoot past open. */
const LID_CLOSED_AT = 0.1;
function lid(since: number) {
  if (since < 0) return 0;
  if (since < 0.1) return easeIn(since / 0.1);
  if (since < 0.15) return 1;
  if (since < 0.37) return 1 - easeOutBack((since - 0.15) / 0.22);
  return 0;
}

/** An ear flick: 60 ms out, 140 ms back. */
function flick(since: number) {
  if (since < 0 || since > 0.2) return 0;
  return since < 0.06 ? since / 0.06 : 1 - easeIn((since - 0.06) / 0.14) ** 0.5;
}

const LOOK: Partial<Record<StateId, Look>> = { writing: "text" };

export type EngineOptions = {
  state: string;
  blinkRate: number;
  gazeRange: number;
  speed: number;
  restartKey: number;
  /** For the 'watch' state: where to look, -1..1 each way (x right, y down) from the face. */
  lookAt?: { x: number; y: number } | null;
  /** For the 'loading' state: real progress 0..1 fills the eyes; without it they fill over time. */
  progress?: number | null;
};

export function useFace(opts: EngineOptions, stage: RefObject<HTMLElement | null>): Face {
  const [face, setFace] = useState<Face>(REST);
  const optsRef = useRef(opts);
  optsRef.current = opts;
  const reduced = useMediaQuery("(prefers-reduced-motion: reduce)");
  // With reduced motion the face is posed once per state instead of animated.
  const poseKey = reduced ? `${opts.state}:${opts.progress ?? ""}` : "";

  useEffect(() => {
    const pointer = { x: 0, y: 0, at: -Infinity };
    const onMove = (e: PointerEvent) => {
      const el = stage.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      pointer.x = Math.max(-1, Math.min(1, (e.clientX - (r.left + r.width / 2)) / (r.width * 0.6)));
      pointer.y = Math.max(-1, Math.min(1, (e.clientY - (r.top + r.height / 2)) / (r.height * 0.6)));
      pointer.at = performance.now() / 1000;
    };
    window.addEventListener("pointermove", onMove);

    const sim: Face = { ...REST, zs: [] };
    const zs: Array<{ id: number; age: number }> = [];
    let target = { x: 0, y: 0 };
    let nextSaccade = 0;
    let nextBlink = 1;
    let blinkAt = -10;
    let pairAt = -1;
    let nextZ = 0;
    let zid = 0;
    let nextToken = 0;
    let lookAt = -1;
    // Ear flicks: a fast flick outward and back, one ear at a time, every few seconds.
    let flickL = -10;
    let flickR = -10;
    let nextFlick = 2.5;
    let tiltL = 0;
    let tiltR = 0;
    let nextLook: Look = "eyes";
    let lastState: StateId | null = null;
    let enteredAt = 0;
    // Scripted moments fire once per visit to a state, whatever the frame rate.
    const fired = new Set<string>();
    const once = (key: string, when: boolean) => {
      if (!when || fired.has(key)) return false;
      fired.add(key);
      return true;
    };

    const blink = (t: number) => {
      blinkAt = t;
      if (Math.random() < 0.12) pairAt = t + 0.45;
    };

    const step = (dtReal: number) => {
      const o = optsRef.current;
      const dt = dtReal * o.speed;
      sim.t += dt;
      const t = sim.t;
      const { id, local, duration } = stateAt(o.state, t);
      if (id !== lastState) {
        lastState = id;
        enteredAt = t;
        fired.clear();
        // Entering a state is a moment: a glance and sometimes a blink go with it. When the eyes
        // change into something else (lines of text), the change happens behind a blink.
        nextSaccade = t + 0.2;
        const look = LOOK[id] ?? "eyes";
        if (look !== nextLook) {
          nextLook = look;
          blinkAt = t;
          pairAt = -1;
          lookAt = t + LID_CLOSED_AT;
        } else if (id !== "resting" && id !== "loading" && id !== "waking" && Math.random() < 0.3) blink(t + 0.08);
        if (id === "reading") sim.text = 0;
        nextBlink = Math.max(nextBlink, t + 1.5);
      }
      if (lookAt > 0 && t >= lookAt) {
        sim.look = nextLook;
        lookAt = -1;
      }
      const since = t - enteredAt;
      // Progress through the state; a state held on its own tab loops at its story length.
      const loop = STORY.find(([sid]) => sid === id)?.[1] ?? 3;
      const p = Number.isFinite(duration) ? local / duration : (since % loop) / loop;
      sim.state = id;
      sim.caption = STATE_LABEL[id];

      // Targets for this moment.
      let gx = 0;
      let gy = 0;
      let squint = 0;
      let happy = 0;
      let size = 1;
      let fill = -1;
      let bracketL = 0;
      let bracketR = 0;
      let asym = 0;
      let sleep = 0;
      const jitter = 0;
      let lift = 0;
      let saccades = false;
      let saccadeGap = 2;
      let blinks = true;
      let gazeRate = 14;
      let sleepRate = 10;
      let glasses = 0;
      let shake = 0;
      let spin = 0;
      let phone = 0;
      let mac = 0;
      let tilt = 0;
      let bob = 0;
      let earH = 1;
      let earTiltL = 0;
      let earTiltR = 0;
      let flicks = true;

      switch (id) {
        case "resting":
          sleep = 1;
          blinks = false;
          gy = 0.35;
          break;
        case "waking": {
          // Slow: still asleep, lids half open and heavy, a drowsy look around, one blink, awake.
          sleep = since < 0.7 ? 1 : since < 2.1 ? 0.5 : 0;
          sleepRate = since < 2.1 ? 3 : 5;
          gazeRate = 4;
          gx = since < 1.2 ? 0 : since < 2.4 ? -0.35 : 0.3;
          gy = since < 2.1 ? 0.3 : 0;
          if (once("awake", since > 1.95)) blinkAt = t;
          blinks = false;
          break;
        }
        case "loading":
          fill = o.progress != null ? Math.max(0, Math.min(1, o.progress)) : Math.min(1, easeIn(p) * 0.15 + p * 0.85);
          gy = 0.55;
          blinks = false;
          if (once("loaded", p > 0.95)) blink(t);
          break;
        case "warming":
          saccades = true;
          saccadeGap = 0.75;
          squint = 0.12;
          break;
        case "reading": {
          // Glasses on, then two lines with four fixations each and a return sweep.
          glasses = 1; // the drawing shrinks the eyes to fit the lenses
          const q = Math.max(0, since - 0.5) / Math.max(1, (Number.isFinite(duration) ? duration : 3.8) - 0.9);
          const line = Math.floor(q * 2) % 2;
          const pos = (q * 2) % 1;
          gx = since < 0.5 ? 0 : pos > 0.9 ? -1 : -1 + Math.floor(pos * 4) * (2 / 3);
          gy = since < 0.5 ? 0 : -0.45 + line * 0.9;
          gazeRate = 18;
          blinks = false;
          break;
        }
        case "thinking": {
          // Daydreaming: eyes up and to the side, drifting slowly as if following a thought,
          // lids a little heavy, the right eye narrower. Same at every size; small sizes get the
          // full room inside the brackets so the look up still reads.
          const full = 1 / Math.max(0.3, o.gazeRange);
          gx = (0.5 + 0.2 * Math.sin(since * 0.75)) * full * 0.9;
          gy = (-0.85 - 0.12 * Math.sin(since * 1.1 + 1)) * full * 0.9;
          gazeRate = 3.5;
          squint = 0.3;
          asym = 1;
          blinks = since > 1.6;
          break;
        }
        case "drafting":
          // Speculative decoding: a draft glance, then verify on the other side.
          gx = Math.floor(since / 0.65) % 2 === 0 ? -0.7 : 0.7;
          gy = 0.1;
          squint = 0.1;
          break;
        case "writing":
          // Eyes become two columns of text; tokens land one by one with a block cursor.
          if (sim.look === "text" && t > nextToken) {
            sim.text += 1;
            nextToken = t + 0.09 + Math.random() * 0.07;
          }
          blinks = false;
          break;
        case "tool": {
          // Calling a tool, as clockwork: the eyes tick round a notch at a time like two meshing
          // gears (left one way, right the other), while the face holds still and attentive.
          const step = 0.55;
          const u = Math.min(1, (since % step) / 0.22);
          spin = 45 * (Math.floor(since / step) + (1 - (1 - u) ** 3));
          size = 0.9;
          gazeRate = 8;
          blinks = false;
          break;
        }
        case "memory": {
          // Tight on memory: wide eyes, squeezed brackets, flat ears, and trembling like a cold
          // cat, fast small shivers of the whole face in bursts.
          size = 1.06;
          bracketL = -6;
          bracketR = -6;
          blinks = false;
          const k = since % 1.15;
          const envelope = k < 0.8 ? Math.sin((k / 0.8) * Math.PI) : 0;
          shake = 1.2 * envelope * Math.sin(t * Math.PI * 2 * 15);
          break;
        }
        case "freeing": {
          // Relief: the shiver fades, a long slow blink like an exhale, the brackets breathe out
          // past their rest and settle.
          const k = since % 3;
          shake = k < 0.4 ? 1.2 * (1 - k / 0.4) * Math.sin(t * Math.PI * 2 * 15) : 0;
          size = k < 0.4 ? 1.06 : 1;
          sleep = k > 0.45 && k < 0.95 ? 1 : 0;
          sleepRate = 7;
          bracketL = k < 0.5 ? -6 : k < 1.3 ? 2.5 : 0;
          bracketR = bracketL;
          blinks = false;
          break;
        }
        case "done":
          happy = 1;
          lift = -2 * Math.abs(Math.sin(since * 4.5)) * Math.max(0, 1 - since / 1.2);
          blinks = false;
          break;
        case "local": {
          // Stays on device: the brackets close into an iPhone for a couple of seconds, widen
          // into a Mac, then open back into the face, with a trusting blink on each screen.
          const loop = Math.floor(since / 7.6);
          const k = since % 7.6;
          phone = k > 0.6 && k < 3.2 ? 1 : 0;
          mac = k >= 3.2 && k < 6.0 ? 1 : 0;
          gy = phone ? 0.15 : 0;
          gx = mac ? (k < 4.6 ? -0.35 : 0.35) : 0;
          gazeRate = 6;
          if (once(`phone-blink-${loop}`, k > 1.9)) blinkAt = t;
          if (once(`mac-blink-${loop}`, k > 5.2)) blinkAt = t;
          if (once(`home-${loop}`, k > 6.9)) blinkAt = t;
          blinks = false;
          break;
        }
        case "watch": {
          // Watching something it's been pointed at (the caret as you type): eyes on it, steady,
          // using the full room inside the brackets, a touch of focus in the lids.
          const full = 1 / Math.max(0.3, o.gazeRange);
          gx = Math.max(-1, Math.min(1, o.lookAt?.x ?? 0)) * full;
          gy = Math.max(-1, Math.min(1, o.lookAt?.y ?? 0.8)) * full;
          gazeRate = 12;
          squint = 0.06;
          break;
        }
        case "grumpy": {
          // Asked to work with nothing to work with: lids drop to a flat stare and the head gives a
          // quick "no" shake, then the eyes go pointedly to what's missing (lookAt, the model picker).
          const full = 1 / Math.max(0.3, o.gazeRange);
          squint = 0.5;
          blinks = false;
          gazeRate = 12;
          shake = since < 0.55 ? 3.2 * Math.sin(since * Math.PI * 2 * 3.6) * (1 - since / 0.55) : 0;
          if (since > 0.6) {
            gx = Math.max(-1, Math.min(1, o.lookAt?.x ?? 0.6)) * full;
            gy = Math.max(-1, Math.min(1, o.lookAt?.y ?? 0.6)) * full;
          }
          break;
        }
        case "follow":
          if (performance.now() / 1000 - pointer.at < 2.5) {
            gx = pointer.x;
            gy = pointer.y;
          } else saccades = true;
          break;
        case "ready": {
          // The empty state, an eight-second loop: a slow blink at you, a glance down at the box
          // where you'd type, ears up, a happy hello with a small hop and brackets opening like
          // arms, then quiet breathing and a look around. A moving cursor always wins.
          const loop = Math.floor(since / 8);
          const k = since % 8;
          gazeRate = 7;
          blinks = false;
          if (once(`slow-${loop}`, k > 0.6)) blinkAt = t;
          // Glances use the full room inside the brackets, whatever the gaze range is set to.
          const full = 1 / Math.max(0.3, o.gazeRange);
          if (k < 1.3) [gx, gy] = [0, 0];
          else if (k < 2.4) {
            // Looking down at the box: eyes drop to the floor of the brackets, upper lids lower.
            [gx, gy] = [-0.15 * full, full];
            squint = 0.3;
            lift = 2.5;
          } else if (k < 3.2) [gx, gy] = [0, 0];
          else if (k < 4.5) {
            [gx, gy] = [0, 0];
            happy = 1;
            const hop = Math.max(0, Math.sin(((k - 3.3) / 0.9) * Math.PI * 2));
            lift = k < 4.2 ? -3.2 * hop : 0;
            bracketL = bracketR = 3;
          } else if (k < 6) {
            lift = Math.sin((k - 4.5) * Math.PI * 1.2) * 0.9;
            if (once(`settle-${loop}`, k > 5.3)) blinkAt = t;
          } else if (k < 7.2) [gx, gy] = [0.9 * full, -0.3 * full];
          else [gx, gy] = [0, 0];
          if (performance.now() / 1000 - pointer.at < 2) {
            gx = pointer.x;
            gy = pointer.y;
            gazeRate = 16;
          }
          break;
        }
        case "welcome": {
          // A happy cat greeting its owner, on a 7.2 s loop of slow, readable beats:
          // 0.0 notices you: eyes wide, ears up, head pops up a little
          // 0.6 meow: ^ ^ eyes and a little hop
          // 1.5 head rubs: the cheek rubs one way, then the other (the face slides and leans)
          // 3.3 purr: a slow blink while the face vibrates softly
          // 4.9 curious: a head tilt with one ear twitching, then settle with a happy blink
          const loop = Math.floor(since / 7.2);
          const k = since % 7.2;
          const bump = (from: number, len: number) =>
            k >= from && k < from + len ? Math.sin(((k - from) / len) * Math.PI) : 0;
          gazeRate = 7;
          blinks = false;
          if (k < 0.6) {
            size = 1.12;
            bob = -2.5 * bump(0, 0.6);
          } else if (k < 1.5) {
            happy = 1;
            bob = -3.2 * bump(0.75, 0.4);
          } else if (k < 3.3) {
            // Two cheek rubs: left, then right. Eased in and out, no jitter.
            const left = bump(1.5, 0.9);
            const right = bump(2.4, 0.9);
            happy = 1;
            shake = -7 * left + 7 * right;
            tilt = -11 * left + 11 * right;
          } else if (k < 4.9) {
            // Purring: eyes close slowly, a soft fast vibration.
            sleep = k > 3.35 && k < 4.7 ? 1 : 0;
            sleepRate = 3.5;
            bob = 0.45 * Math.sin(t * Math.PI * 2 * 22) * bump(3.3, 1.6);
          } else if (k < 6.2) {
            tilt = 10;
            gx = 0.25;
            gy = -0.15;
          } else if (once(`welcome-blink-${loop}`, k > 6.5)) blinkAt = t;
          if (performance.now() / 1000 - pointer.at < 2) {
            gx = pointer.x;
            gy = pointer.y;
            gazeRate = 16;
          }
          break;
        }
        default:
          saccades = true;
          saccadeGap = 2;
      }

      // Ears, per state. Positive tilt leans an ear outward; negative leans it forward.
      switch (id) {
        case "resting":
          earH = 0.84;
          earTiltL = earTiltR = 9;
          break;
        case "waking":
          earH = since < 2.1 ? 0.86 : 1.06;
          earTiltL = earTiltR = since < 2.1 ? 8 : -3;
          if (once("ear-awake", since > 2.15)) flickR = t;
          break;
        case "reading":
          earTiltL = earTiltR = -4;
          break;
        case "thinking":
          earTiltR = 22;
          earTiltL = -2;
          break;
        case "drafting":
          earTiltL = earTiltR = 0;
          earTiltL += gx < 0 ? -8 : 6;
          earTiltR += gx > 0 ? -8 : 6;
          break;
        case "tool":
          earH = 1.06;
          earTiltL = earTiltR = -6;
          break;
        case "grumpy":
          // Ears pinned back and out: annoyed, not scared.
          earH = 0.72;
          earTiltL = earTiltR = 30;
          flicks = false;
          break;
        case "memory":
          // Airplane ears: flat and back while it is tight.
          earH = 0.6;
          earTiltL = earTiltR = 38;
          flicks = false;
          break;
        case "freeing": {
          const k = since % 3;
          earH = k < 0.5 ? 0.6 : k < 1.3 ? 1.1 : 1;
          earTiltL = earTiltR = k < 0.5 ? 38 : 0;
          if (once(`ear-relief-${Math.floor(since / 3)}`, k > 1.0)) flickL = t;
          break;
        }
        case "done":
          earH = 1.12;
          earTiltL = earTiltR = -6;
          break;
        case "local": {
          // Real cat ears are mostly still: on each device they sit a touch forward, flick once
          // shortly after it appears, and on the Mac one ear turns slightly toward where it looks.
          const loop = Math.floor(since / 7.6);
          const k = since % 7.6;
          flicks = false;
          earTiltL = earTiltR = -3;
          if (k >= 4.6 && k < 6.0) earTiltR = -8;
          if (once(`local-phone-${loop}`, k > 1.6)) flickR = t;
          if (once(`local-mac-${loop}`, k > 4.0)) flickL = t;
          break;
        }
        case "welcome": {
          const loop = Math.floor(since / 7.2);
          const k = since % 7.2;
          const bump = (from: number, len: number) =>
            k >= from && k < from + len ? Math.sin(((k - from) / len) * Math.PI) : 0;
          // Up and forward on seeing you; during each cheek rub the ear on that side folds back;
          // relaxed and a little out while purring; one ear forward with the curious tilt.
          earH = k < 1.5 ? 1.18 : k >= 3.3 && k < 4.9 ? 0.92 : 1;
          earTiltL = k < 1.5 ? -7 : k < 3.3 ? 26 * bump(1.5, 0.9) : k < 4.9 ? 9 : 0;
          earTiltR = k < 1.5 ? -7 : k < 3.3 ? 26 * bump(2.4, 0.9) : k < 4.9 ? 9 : k < 6.2 ? -10 : 0;
          if (once(`welcome-pop-l-${loop}`, k > 0.05)) flickL = t;
          if (once(`welcome-pop-r-${loop}`, k > 0.15)) flickR = t;
          if (once(`welcome-curious-${loop}`, k > 5.3)) flickR = t;
          if (once(`welcome-settle-${loop}`, k > 6.8)) flickL = t;
          flicks = false;
          break;
        }
        case "ready": {
          const loop = Math.floor(since / 8);
          const k = since % 8;
          // Ears perk before the hello and stay up through it, with one flick on the way.
          earH = k > 2.5 && k < 4.6 ? 1.12 : 1;
          earTiltL = earTiltR = k > 2.5 && k < 4.6 ? -5 : 0;
          if (once(`ready-flick-${loop}`, k > 2.55)) flickL = t;
          if (once(`ready-flick-r-${loop}`, k > 2.75)) flickR = t;
          break;
        }
        default:
      }
      if (flicks && t > nextFlick) {
        if (Math.random() < 0.5) flickL = t;
        else flickR = t;
        // Sometimes the other ear answers a beat later.
        if (Math.random() < 0.2) {
          if (flickL === t) flickR = t + 0.18;
          else flickL = t + 0.18;
        }
        nextFlick = t + (id === "resting" ? 7 : 3.5) + Math.random() * 5;
      }

      if (saccades && t > nextSaccade) {
        const a = Math.random() * Math.PI * 2;
        const r = Math.sqrt(Math.random());
        target = { x: Math.cos(a) * r, y: Math.sin(a) * r * 0.7 };
        nextSaccade = t + saccadeGap * (0.4 + Math.random() * 1.2);
        if (Math.random() < 0.15) blink(t);
      }
      if (saccades) {
        gx = target.x;
        gy = target.y;
      }

      const range = o.gazeRange;
      sim.gx = approach(sim.gx, gx * range, gazeRate, dt);
      sim.gy = approach(sim.gy, gy * range, gazeRate, dt);
      sim.turn = approach(sim.turn, gx * range, 6, dt);
      sim.squint = approach(sim.squint, squint, 6, dt);
      sim.glasses = approach(sim.glasses, glasses, 7, dt);
      sim.earH = approach(sim.earH, earH, 9, dt);
      tiltL = approach(tiltL, earTiltL, 9, dt);
      tiltR = approach(tiltR, earTiltR, 9, dt);
      sim.earL = tiltL + 26 * flick(t - flickL);
      sim.earR = tiltR + 26 * flick(t - flickR);
      sim.shake = shake;
      sim.tilt = approach(sim.tilt, tilt, 7, dt);
      sim.bob = bob;
      sim.phone = approach(sim.phone, phone, 6, dt);
      sim.mac = approach(sim.mac, mac, 6, dt);
      // Gears turn only while calling a tool; after, the eyes settle to the nearest upright turn.
      sim.spin = id === "tool" ? spin : approach(sim.spin, Math.round(sim.spin / 180) * 180, 8, dt);
      sim.happy = approach(sim.happy, happy, 10, dt);
      sim.size = approach(sim.size, size, id === "memory" ? 6 : 9, dt);
      sim.asym = approach(sim.asym, asym, 6, dt);
      sim.sleep = approach(sim.sleep, sleep, id === "resting" ? 1.8 : sleepRate, dt);
      sim.bracketL = approach(sim.bracketL, bracketL, 9, dt);
      sim.bracketR = approach(sim.bracketR, bracketR, 9, dt);
      sim.jitter = jitter ? (Math.random() - 0.5) * jitter : approach(sim.jitter, 0, 20, dt);
      sim.lift = lift ? lift : approach(sim.lift, 0, 20, dt);
      sim.fill = fill;

      if (blinks && t > nextBlink) {
        blink(t);
        nextBlink = t + (3 + Math.random() * 4) / Math.max(0.1, o.blinkRate);
      }
      if (pairAt > 0 && t > pairAt) {
        blinkAt = t;
        pairAt = -1;
      }
      sim.blinkL = lid(t - blinkAt);
      sim.blinkR = lid(t - blinkAt - 0.012);

      // Resting: small squares drift up, one every ~1.2 s.
      for (const z of zs) z.age += dt;
      while (zs.length && zs[0]!.age > 3) zs.shift();
      if (id === "resting" && sim.sleep > 0.7 && t > nextZ) {
        zs.push({ id: (zid += 1), age: 0 });
        nextZ = t + 1.2;
      }
      sim.zs = zs.map((z) => ({ ...z }));
      sim.breath = Math.sin((t / 3.6) * Math.PI * 2);
    };

    let frame = 0;
    let last = performance.now();
    if (reduced) {
      for (let i = 0; i < 90; i += 1) step(1 / 30);
      setFace({ ...sim, blinkL: 0, blinkR: 0 });
      return () => window.removeEventListener("pointermove", onMove);
    }
    const tick = (now: number) => {
      step(Math.min(now - last, 64) / 1000);
      last = now;
      setFace({ ...sim });
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("pointermove", onMove);
    };
  }, [stage, opts.restartKey, reduced, poseKey]);

  return face;
}
