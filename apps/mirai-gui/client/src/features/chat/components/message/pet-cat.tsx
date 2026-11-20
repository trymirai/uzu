import { type PointerEvent, type RefObject, useEffect, useRef, useState } from "react";

/*
 * Petting uzu on the new-chat screen. Over the cat the cursor becomes a small hand; stroking it
 * (back and forth across its head) or clicking it makes it happy: ^ ^ eyes, the face following the
 * hand with a light purr, a heart
 * floating up from each stroke, and a purring line in place of the greeting's second line.
 *
 * Decided from a prototype with three directions: hearts (this), a hidden stroke with no visible
 * reward (too easy to never find), and a "Pet uzu" tooltip with press-and-hold (too deliberate for
 * something meant to be playful). The purring line came from the tooltip version.
 *
 * The motion was a second prototype: follow-only (alive but didn't read as petting), purr (this),
 * and a squish on every stroke (too busy). A slow lean and sway before it felt sluggish, and a
 * tremble of a pixel or more looked like shaking; the buzz here is 0.55px.
 */

// The Lucide "hand", black on a white halo so it reads on light and dark; 22px, hotspot in the palm.
const HAND_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke-linecap="round" stroke-linejoin="round"><g stroke="#fff" stroke-width="4.2"><path d="M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2"/><path d="M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2"/><path d="M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8"/><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/></g><g stroke="#111" stroke-width="1.8" fill="#fff"><path d="M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2"/><path d="M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2"/><path d="M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8"/><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/></g></svg>`;
export const PET_CURSOR = `url("data:image/svg+xml,${encodeURIComponent(HAND_SVG)}") 10 11, pointer`;

/** What uzu says while you pet it, in place of the greeting's second line. */
export const PETTING_LINES: readonly string[] = [
  "Prrr… that's the spot",
  "Mrrp. Right behind the ears",
  "Purring at full power",
  "Okay, five more minutes",
  "Best part of my day",
  "Don't stop now",
  "You've done this before",
  "Prrr… I could get used to this",
];

/** A line for a new round of petting, never the same as the last one. */
export function choosePettingLine(last: number, random: () => number = Math.random): number {
  const next = Math.floor(random() * PETTING_LINES.length);
  return next === last ? (next + 1) % PETTING_LINES.length : next;
}

const reducedMotion = () =>
  typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// A stroke is the pointer turning back after travelling at least this far; petting lasts this long
// after the last one.
const STROKE_REACH = 14;
const PETTING_HOLD_MS = 750;
const HEART_LIFE_MS = 1100;

export type Heart = { id: number; x: number; drift: number; size: number };

/**
 * Petting for an element holding the cat: pointer handlers to spread on it, whether it's being
 * petted, the hearts in the air, and `pet` for a click or a key. `faceRef` is the element that
 * follows the hand and purrs while petting.
 */
export function useCatPetting(catRef: RefObject<HTMLElement>, faceRef: RefObject<HTMLElement>) {
  const [lastStrokeAt, setLastStrokeAt] = useState(0);
  const [, setTick] = useState(0);
  const [hearts, setHearts] = useState<Heart[]>([]);
  const [line, setLine] = useState(-1);
  const heartId = useRef(0);
  const track = useRef<{ x: number; dir: -1 | 0 | 1; from: number } | null>(null);
  const petting = lastStrokeAt > 0 && performance.now() - lastStrokeAt < PETTING_HOLD_MS;

  // Re-check "still petting?" a few times a second while a stroke is recent.
  useEffect(() => {
    if (!lastStrokeAt) return;
    const timer = setInterval(() => {
      setTick((n) => n + 1);
      if (performance.now() - lastStrokeAt > PETTING_HOLD_MS) clearInterval(timer);
    }, 120);
    return () => clearInterval(timer);
  }, [lastStrokeAt]);

  // Hearts leave on their own.
  useEffect(() => {
    if (!hearts.length) return;
    const timer = setTimeout(() => setHearts((all) => all.slice(1)), HEART_LIFE_MS);
    return () => clearTimeout(timer);
  }, [hearts]);

  // How it moves while petted, from the bottom of the face like a head on a neck: it follows the
  // cursor across its head (a quick, springy tilt and slide) and purrs (a light, fast buzz with a
  // slow breath). It all settles back to still when the petting stops.
  const live = useRef({ petting: false, x: 0, over: false });
  live.current.petting = petting;
  useEffect(() => {
    if (reducedMotion()) return;
    const angle = { x: 0, v: 0 };
    const shift = { x: 0, v: 0 };
    // A damped spring toward a target: stiff enough to keep up with the hand, a touch of overshoot.
    const spring = (s: { x: number; v: number }, target: number, dt: number) => {
      s.v += (-220 * (s.x - target) - 15 * s.v) * dt;
      s.x += s.v * dt;
    };
    let level = 0;
    let last = performance.now();
    let raf = 0;
    const tick = (t: number) => {
      const dt = Math.min(0.033, (t - last) / 1000);
      last = t;
      const { petting: on, over, x } = live.current;
      level += ((on ? 1 : 0) - level) * 0.15;
      const follow = on && over ? x : 0;
      spring(angle, follow * 5, dt);
      spring(shift, follow * 2, dt);
      const el = faceRef.current;
      if (el) {
        if (level < 0.01 && Math.abs(angle.x) < 0.01 && Math.abs(shift.x) < 0.01) el.style.transform = "";
        else {
          const buzz = Math.sin(t * 0.14) * 0.55 * level;
          const breathe = 1 + Math.sin(t * 0.0075) * 0.02 * level;
          el.style.transformOrigin = "50% 90%";
          el.style.transform = `translateX(${shift.x.toFixed(2)}px) rotate(${angle.x.toFixed(2)}deg) translateY(${buzz.toFixed(2)}px) scale(${breathe.toFixed(4)})`;
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [faceRef]);

  const pet = (clientX?: number) => {
    if (!petting) setLine((last) => choosePettingLine(last));
    setLastStrokeAt(performance.now());
    const box = catRef.current?.getBoundingClientRect();
    if (!box) return;
    // From the stroke's spot along the head, or the middle for a click or a key.
    const x = clientX === undefined ? box.width / 2 : Math.min(box.width - 16, Math.max(16, clientX - box.left));
    heartId.current += 1;
    const id = heartId.current;
    // A gentle, varied drift and size without randomness, so no two neighbours match.
    setHearts((all) => [...all.slice(-11), { id, x, drift: ((id * 37) % 21) - 10, size: 12 + ((id * 13) % 7) }]);
  };

  const onPointerMove = (event: PointerEvent) => {
    const x = event.clientX;
    // Where the hand is across the cat (-1..1), for the face to follow.
    const box = catRef.current?.getBoundingClientRect();
    if (box) {
      live.current.over = true;
      live.current.x = Math.max(-1, Math.min(1, (x - (box.left + box.width / 2)) / (box.width / 2)));
    }
    const prev = track.current;
    if (!prev) {
      track.current = { x, dir: 0, from: x };
      return;
    }
    const dir = x > prev.x ? 1 : x < prev.x ? -1 : prev.dir;
    if (dir !== prev.dir && prev.dir !== 0 && Math.abs(prev.x - prev.from) >= STROKE_REACH) {
      pet(x);
      track.current = { x, dir, from: prev.x };
      return;
    }
    track.current = { x, dir, from: dir !== prev.dir ? prev.x : prev.from };
  };

  return {
    petting,
    hearts,
    line: petting && line >= 0 ? PETTING_LINES[line] : null,
    pet,
    handlers: {
      onPointerMove,
      onPointerLeave: () => {
        track.current = null;
        live.current.over = false;
      },
      onClick: (event: PointerEvent | React.MouseEvent) => pet(event.clientX),
      onKeyDown: (event: React.KeyboardEvent) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        pet();
      },
    },
  };
}

/** One heart: rises from the cat's head with a soft pop, drifts a little, blurs away. */
export function CatHeart({ heart }: { heart: Heart }) {
  const ref = useRef<SVGSVGElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (reducedMotion()) {
      el.animate([{ opacity: 0.9 }, { opacity: 0 }], { duration: 700, easing: "ease-out", fill: "forwards" });
      return;
    }
    el.animate(
      [
        { transform: "translate(-50%, 0) scale(0.6)", opacity: 0, filter: "blur(2px)" },
        {
          transform: `translate(calc(-50% + ${heart.drift * 0.3}px), -14px) scale(1.05)`,
          opacity: 1,
          filter: "blur(0)",
          offset: 0.22,
        },
        { transform: `translate(calc(-50% + ${heart.drift}px), -54px) scale(0.9)`, opacity: 0, filter: "blur(3px)" },
      ],
      { duration: HEART_LIFE_MS, easing: "cubic-bezier(0.32, 0.72, 0, 1)", fill: "forwards" },
    );
  }, [heart.drift]);
  return (
    <svg
      ref={ref}
      aria-hidden
      width={heart.size}
      height={heart.size}
      viewBox="0 0 24 24"
      className="absolute top-3"
      style={{ left: heart.x, opacity: 0 }}
    >
      <path
        d="M12 21s-7.5-4.6-9.6-9.2C.9 8.5 3 4.5 6.7 4.5c2.1 0 3.6 1.2 4.3 2.4h2c.7-1.2 2.2-2.4 4.3-2.4 3.7 0 5.8 4 4.3 7.3C19.5 16.4 12 21 12 21z"
        fill="#ff7a9c"
      />
    </svg>
  );
}
