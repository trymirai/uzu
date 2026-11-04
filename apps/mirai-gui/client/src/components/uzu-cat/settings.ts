import { earSize, type Settings } from "./face";

/** The uzu cat: soft and rounded, pill eyes, rounded brackets with round ends, set-in cat ears
 *  with slightly rounded tips, v when happy. */
export const FAVOURITE: Settings = {
  state: "story",
  detail: "full",
  brackets: "square",
  eyeShape: "square",
  eyeRadius: 0.5,
  happyShape: "vee",
  ears: "cat",
  earBase: false,
  earScale: 0.8,
  eyeAspect: 0.54,
  weight: 5,
  eyeSize: 38,
  gap: 30,
  spread: 50,
  bracketHeight: 64,
  gazeRange: 0.7,
  blinkRate: 1,
  speed: 1,
  earTip: 0.3,
  earInset: 11,
  earRise: 1,
  earMotion: true,
  earArm: true,
  eyeSymbol: "block",
  eyelids: false,
  lidSize: 0.3,
  cornerRadius: 10,
  roundEnds: true,
  pxCols: 5,
  pxRows: 7,
  pxTop: 3,
  pxBottom: 3,
  pxSide: 1,
  pxGap: 0,
  pxRound: 0,
  earRadius: 2,
};

/**
 * The same face, adapted to a size in pixels. Below 96 px every line is kept to at least
 * ~1.25-1.5 px, every gap (eye to eye, eye to bracket) to at least ~1 px, and below 28 px the
 * drawing switches to its tiny version: solid ears and two lines of text.
 */
export function forSize(s: Settings, px: number): Settings {
  const unit = 160 / px; // viewBox units per pixel
  if (px >= 96) return { ...s, detail: "full", unit };
  const tiny = px < 28;
  const weight = Math.max(s.weight, (tiny ? 1.5 : 1.25) * unit);
  // Eye to eye (horizontal sizes go through the eye's width).
  const aspect = s.eyeAspect ?? 1;
  // Only when the design has a gap at all: eyes set to touch or overlap stay that way.
  let eyeSize = s.eyeSize;
  if (s.gap > s.eyeSize * aspect)
    eyeSize = Math.max(unit * 3, Math.min(eyeSize, (s.gap - (tiny ? 1.4 : 1.2) * unit) / aspect));
  // Eye to spine.
  const sideRoom = s.spread - weight / 2 - (s.gap / 2 + (eyeSize * aspect) / 2);
  if (sideRoom < unit) eyeSize -= (2 * (unit - sideRoom)) / aspect;
  // Eye to the bracket arms: grow the brackets rather than shrink the eyes further.
  const bracketHeight = Math.max(s.bracketHeight, 2 * (weight + eyeSize / 2 + unit));
  const adapted: Settings = {
    ...s,
    weight,
    eyeSize,
    bracketHeight,
    detail: tiny ? "tiny" : "medium",
    unit,
    ears: s.ears === "none" ? "none" : tiny ? "cat-solid" : s.ears,
  };
  // Keep the ears inside the box: trim the brackets if the ear tips would leave it.
  if (adapted.ears !== "none") {
    const { rise } = earSize(adapted);
    const over = 80 - bracketHeight / 2 - rise * 0.58 - weight - 2;
    if (over < 0) adapted.bracketHeight = Math.max(2 * (weight + eyeSize / 2), bracketHeight + over * 1.2);
  }
  return adapted;
}
