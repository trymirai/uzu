import type { ReactNode } from "react";
import type { Face } from "./engine";

/*
 * The uzu cat: a matrix face, [ eyes ] in square brackets, with cat ears on the bracket corners.
 * It lives in a 160 box. The engine (engine.ts) decides what the face does; this file draws it.
 */

export type Settings = {
  state: string;
  brackets: string;
  eyeShape: string;
  weight: number;
  eyeSize: number;
  gap: number;
  spread: number;
  bracketHeight: number;
  blinkRate: number;
  gazeRange: number;
  speed: number;
  /** Eye corner radius as a share of the eye's short side. */
  eyeRadius: number;
  /** Happy eyes: 'caret' (^), 'vee' (v) or 'bar' (a flat squint). */
  happyShape: string;
  /** Cat ears on the bracket corners: 'none', 'cat' (outline) or 'cat-solid'. */
  ears: string;
  /** How much detail survives: 'full', or simplified drawings for 'medium' and 'tiny' sizes. */
  detail?: string;
  /** Eye width as a share of its height (1 = square). */
  eyeAspect?: number;
  /** Ear size multiplier. */
  earScale?: number;
  /** Outline ears: draw the bracket's top arm under the ear (closed) or leave it open. */
  earBase?: boolean;
  /** Ear height as a share of its width. */
  earRise?: number;
  /** Where the tip sits over the base: 0 straight above the outer foot, 1 above the inner foot. */
  earTip?: number;
  /** How far the ear sits in from the bracket's corner, in box units. */
  earInset?: number;
  /** Ears react to the states (flatten, perk up, flick); off keeps them still. */
  earMotion?: boolean;
  /** Corner radius for the brackets, on the line's centre, in box units. */
  cornerRadius?: number;
  /** Corner radius for the ears (tip and feet); falls back to the bracket radius when unset. */
  earRadius?: number;
  /** Round the open ends of the bracket arms instead of cutting them square. */
  roundEnds?: boolean;
  /** Box units per screen pixel at the size being drawn (set by forSize); keeps gaps ≥ ~1 px. */
  unit?: number;
  /** The top line runs on past the ear (true) or stops where the ear ends (false). */
  earArm?: boolean;
  /** Eye symbol: 'block' (solid), 'zero' (outlined), 'slashed' (0 with a slash), 'dotted' (0 with a dot). */
  eyeSymbol?: string;
  /** Separate eyelids: a lid line over each eye, with a gap below it. */
  eyelids?: boolean;
  /** How much of the eye the lids cover at rest, 0..0.7. */
  lidSize?: number;
  /** Pixel zeros: grid columns and rows, top and bottom bar widths, side thickness (all in
   *  pixels), and the gap between pixels as a share of a pixel (0 = solid, >0 = LED matrix). */
  pxCols?: number;
  pxRows?: number;
  pxTop?: number;
  pxBottom?: number;
  pxSide?: number;
  pxGap?: number;
  /** Pixel zeros: corner radius of each pixel, as a share of the pixel (0..0.5). */
  pxRound?: number;
  /** Internal, set during the device morph: a fixed ear base width, and how far the line past
   *  the ear has grown toward the middle (0..1) when `earArm` is off. */
  earB?: number;
  armGrow?: number;
};

export type CharacterProps = {
  f: Face;
  s: Settings;
  ink: string;
  paper: string;
  /** Unique per rendered SVG, for clip ids. */
  uid?: string;
  /** Fill the inside of the outline (head and ears) with this colour, e.g. for a sticker's die-cut. */
  backing?: string;
};
export type Character = {
  id: string;
  name: string;
  draw: (p: CharacterProps) => ReactNode;
};

const f1 = (n: number) => Math.round(n * 10) / 10;

/* ------------------------------------------------------------------ rounded corners */

type Pt = [number, number];

/**
 * A polyline with its corners filleted at radius `r` (on the line's centre), so a stroke keeps an
 * even weight round the bend: outer edge r + w/2, inner edge r - w/2. Each fillet takes at most
 * half of a shared segment, so tight corners (an ear's tip) shrink their radius instead of
 * overlapping. `roundAt` picks which vertices get rounded.
 */
function roundedPath(
  pts: Pt[],
  r: number | ((i: number) => number),
  closed = false,
  roundAt: (i: number) => boolean = () => true,
): string {
  const n = pts.length;
  const p = (q: Pt) => `${f1(q[0])} ${f1(q[1])}`;
  // One radius for every corner, or a radius per vertex.
  const radiusAt = typeof r === "function" ? r : () => r;
  if (typeof r === "number" && r <= 0.05) return `M${pts.map(p).join(" L")}${closed ? " Z" : ""}`;
  let d = "";
  for (let i = 0; i < n; i += 1) {
    const P = pts[i]!;
    const end = !closed && (i === 0 || i === n - 1);
    const A = pts[(i - 1 + n) % n]!;
    const B = pts[(i + 1) % n]!;
    const la = Math.hypot(A[0] - P[0], A[1] - P[1]);
    const lb = Math.hypot(B[0] - P[0], B[1] - P[1]);
    const plain = () => {
      d += `${d ? " L" : "M"}${p(P)}`;
    };
    const ri = radiusAt(i);
    if (end || !roundAt(i) || ri <= 0.05 || la < 1e-6 || lb < 1e-6) {
      plain();
      continue;
    }
    const da: Pt = [(A[0] - P[0]) / la, (A[1] - P[1]) / la];
    const db: Pt = [(B[0] - P[0]) / lb, (B[1] - P[1]) / lb];
    const angle = Math.acos(Math.max(-1, Math.min(1, da[0] * db[0] + da[1] * db[1])));
    if (angle > Math.PI - 0.01) {
      plain(); // straight through
      continue;
    }
    const half = Math.tan(angle / 2);
    const maxA = closed || i - 1 > 0 ? la / 2 : la;
    const maxB = closed || i + 1 < n - 1 ? lb / 2 : lb;
    const tan = Math.min(ri / half, maxA, maxB);
    const rr = tan * half;
    const P1: Pt = [P[0] + da[0] * tan, P[1] + da[1] * tan];
    const P2: Pt = [P[0] + db[0] * tan, P[1] + db[1] * tan];
    // Turning clockwise on screen (y down) is a positive cross product: sweep 1.
    const sweep = (P[0] - A[0]) * (B[1] - P[1]) - (P[1] - A[1]) * (B[0] - P[0]) > 0 ? 1 : 0;
    d += `${d ? " L" : "M"}${p(P1)} A${f1(rr)} ${f1(rr)} 0 0 ${sweep} ${p(P2)}`;
  }
  return closed ? `${d} Z` : d;
}

/* ------------------------------------------------------------------ brackets */

function bracket(
  style: string,
  side: -1 | 1,
  cx: number,
  h: number,
  w: number,
  ink: string,
  key: string,
  radius = 0,
  roundEnds = false,
  armLen?: number,
) {
  const top = 80 - h / 2;
  const bottom = 80 + h / 2;
  const x = cx; // the bracket's spine
  const arm = (armLen ?? Math.max(w * 1.6, h * 0.2)) * -side; // arms point inward
  const common = {
    stroke: ink,
    strokeWidth: w,
    fill: "none",
    strokeLinejoin: "miter" as const,
    strokeMiterlimit: 10,
  };
  if (style === "none") return null;
  if (style === "bars") return <path key={key} d={`M${f1(x)} ${f1(top)} V${f1(bottom)}`} {...common} />;
  if (style === "round") {
    return (
      <path
        key={key}
        d={`M${f1(x + arm)} ${f1(top)} Q${f1(x - arm * 0.6)} 80 ${f1(x + arm)} ${f1(bottom)}`}
        {...common}
        strokeLinecap="round"
      />
    );
  }
  if (style === "angle")
    return <path key={key} d={`M${f1(x + arm)} ${f1(top)} L${f1(x)} 80 L${f1(x + arm)} ${f1(bottom)}`} {...common} />;
  if (style === "curly") {
    const a = arm * 0.9;
    return (
      <path
        key={key}
        d={`M${f1(x + a)} ${f1(top)} C${f1(x)} ${f1(top)} ${f1(x + a * 0.15)} ${f1(top + h * 0.12)} ${f1(x + a * 0.15)} ${f1(80 - h * 0.14)} C${f1(x + a * 0.15)} ${f1(80 - h * 0.04)} ${f1(x - a * 0.55)} 80 ${f1(x - a * 0.55)} 80 C${f1(x - a * 0.55)} 80 ${f1(x + a * 0.15)} ${f1(80 + h * 0.04)} ${f1(x + a * 0.15)} ${f1(80 + h * 0.14)} C${f1(x + a * 0.15)} ${f1(bottom - h * 0.12)} ${f1(x)} ${f1(bottom)} ${f1(x + a)} ${f1(bottom)}`}
        {...common}
        strokeLinecap="round"
      />
    );
  }
  // Square: a spine with short arms, square-ended like matrix notation.
  const pts: Pt[] = [
    [x + arm, top + w / 2],
    [x, top + w / 2],
    [x, bottom - w / 2],
    [x + arm, bottom - w / 2],
  ];
  return <path key={key} d={roundedPath(pts, radius)} {...common} strokeLinecap={roundEnds ? "round" : "butt"} />;
}

/** How far the face sits below centre to look optically centred (ears add height on top). */
export function faceShift(s: Settings) {
  return s.ears !== "none" && s.brackets === "square" ? earSize(s).rise * 0.42 : 0;
}

/* ------------------------------------------------------------------ cat ears */

/**
 * Cat ear proportions, on the stroke's centreline. The base runs from the spine inward; at rest the
 * tip sits straight above the spine, so the outer edge is the spine carried on up and only the
 * inner edge is angled.
 */
export function earSize(s: Settings) {
  // Small drawings carry heavier lines, so the ear's minimum is held back to keep it an ear, not a
  // block: the face stays the main shape.
  const k = s.earScale ?? 1;
  const B =
    s.earB ??
    k *
      (s.detail === "tiny"
        ? Math.max(s.weight * 2.2, s.bracketHeight * 0.34)
        : s.detail === "medium"
          ? Math.max(s.weight * 3, s.bracketHeight * 0.4)
          : Math.max(s.weight * 3.6, s.bracketHeight * 0.42));
  return {
    B,
    rise: B * (s.earRise ?? 0.8),
    tip: s.earTip ?? 0,
    inset: s.earInset ?? 0,
  };
}

/**
 * A square bracket wearing a cat ear, drawn as one stroke so every join is a true miter: up the
 * spine and straight on to the ear's tip, down its angled inner edge, then on along the top arm,
 * which also runs under the ear. The solid version fills the ear too.
 */
function catBracket(
  side: -1 | 1,
  x: number,
  h: number,
  f: Face,
  s: Settings,
  ink: string,
  armLen?: number,
  backing?: string,
) {
  const w = s.weight;
  const topC = 80 - h / 2 + w / 2;
  const botC = 80 + h / 2 - w / 2;
  const arm = armLen ?? Math.max(w * 1.6, h * 0.2);
  const ear = earSize(s);
  const { rise, tip: tipAt } = ear;
  // While the brackets close into a device (armLen given), nothing on the top line may reach past
  // the middle: the folded ear and the arm's end stop at the centre, where the other half takes
  // over, so the top edge ends exactly at the device's sides instead of overhanging like a shelf.
  const reach = armLen !== undefined ? Math.abs(80 - x) : Infinity;
  const inset = Math.min(ear.inset, reach);
  const B = Math.min(ear.B, reach - inset);
  const inward = -side;
  const foot = x + inward * inset; // the ear's outer foot on the top arm
  const inner = foot + inward * B;
  // The tip pivots on the outer foot. At rest it sits `tipAt` of the way along the base; the
  // engine leans it out or forward and makes it shorter or taller.
  const moving = s.earMotion !== false;
  const lean = moving ? ((side < 0 ? f.earL : f.earR) * Math.PI) / 180 : 0;
  const u0 = tipAt * B;
  const h0 = rise * (moving ? f.earH : 1);
  const u = u0 * Math.cos(lean) - h0 * Math.sin(lean);
  const v = u0 * Math.sin(lean) + h0 * Math.cos(lean);
  const tip: [number, number] = [foot + inward * u, topC - v];
  const solid = s.ears === "cat-solid" || s.detail === "tiny";
  const baseOn = solid || s.earBase !== false;
  // The line past the ear: on by default; off ends the line where the ear ends. On a device it
  // grows back in toward the middle so the outline still closes.
  const afterOff =
    s.earArm === false ? (s.armGrow ?? 0) * Math.max(0, (Number.isFinite(reach) ? reach : 0) - B - inset) : 0;
  const hasArm = s.earArm !== false || afterOff > 0.05;
  const topArmEnd =
    s.earArm !== false
      ? x + inward * Math.min(reach, B + inset + Math.max(w * 1.2, arm - B - inset))
      : inner + inward * afterOff;
  const r = s.cornerRadius ?? 0;
  // The ear has its own radius (tip and both feet); the bracket's corners keep theirs. With no
  // inset the top corner is the ear's outer foot, so it takes the ear's radius.
  const earR = s.earRadius ?? r;
  const common = {
    stroke: ink,
    strokeWidth: w,
    strokeLinejoin: "miter" as const,
    strokeMiterlimit: 12,
    strokeLinecap: s.roundEnds ? ("round" as const) : ("butt" as const),
  };
  // The contour: bottom arm, spine, (along the top arm to the ear), up to the tip, down, top arm.
  // With no line past the ear and a base under it, the stroke turns at the inner foot and runs
  // back along the base, so that corner is a true miter instead of a cut end.
  const tail: Pt[] = hasArm ? [[topArmEnd, topC]] : baseOn ? [[foot, topC]] : [];
  const pts: Pt[] = [
    [x + inward * arm, botC],
    [x, botC],
    [x, topC],
    ...(inset > 0.05 ? [[foot, topC] as Pt] : []),
    tip,
    [inner, topC],
    ...tail,
  ];
  const earStart = inset > 0.05 ? 3 : 2;
  // The run under the ear starts where the top corner's rounding ends, so it never squares a
  // rounded corner back up, and runs on to the arm's end so the ear's feet stay filled below.
  const cornerAt = inset > 0.05 ? Math.min(r, inset / 2) : 0;
  const baseFrom = x + inward * cornerAt;
  const base = baseOn && hasArm ? ` M${f1(baseFrom)} ${f1(topC)} H${f1(topArmEnd)}` : "";
  return (
    <g key={side}>
      {backing ? (
        <path d={roundedPath([[foot, topC], tip, [inner, topC]], earR, true, (i) => i === 1)} fill={backing} />
      ) : null}
      {solid ? (
        <path d={roundedPath([[foot, topC], tip, [inner, topC]], earR, true, (i) => i === 1)} fill={ink} />
      ) : null}
      {/* One path: the contour plus the arm under the ear as a second run inside the contour's own
          band, so the joins stay true miters (or even fillets) and nothing meets on a seam. */}
      <path d={`${roundedPath(pts, (i) => (i >= earStart ? earR : r))}${base}`} fill="none" {...common} />
    </g>
  );
}

function Brackets({
  f,
  s,
  ink,
  spread,
  height,
  left = 0,
  right = 0,
  arm,
  backing,
}: {
  f: Face;
  s: Settings;
  ink: string;
  spread: number;
  height: number;
  left?: number;
  right?: number;
  arm?: number;
  backing?: string;
}) {
  const h = height * (1 + 0.012 * f.breath);
  const cat = s.ears !== "none" && s.brackets === "square";
  // Small sizes exaggerate the bracket moves so they still read at a few pixels.
  const k = s.detail === "tiny" ? 1.8 : s.detail === "medium" ? 1.4 : 1;
  const xl = 80 - spread - left * k;
  const xr = 80 + spread + right * k;
  if (cat) {
    return (
      <>
        {backing ? (
          <rect
            x={f1(xl - s.weight / 2)}
            y={f1(80 - h / 2)}
            width={f1(xr - xl + s.weight)}
            height={f1(h)}
            rx={f1((s.cornerRadius ?? 0) + s.weight / 2)}
            fill={backing}
          />
        ) : null}
        {catBracket(-1, xl, h, f, s, ink, arm, backing)}
        {catBracket(1, xr, h, f, s, ink, arm, backing)}
      </>
    );
  }
  return (
    <>
      {backing ? (
        <rect
          x={f1(xl - s.weight / 2)}
          y={f1(80 - h / 2)}
          width={f1(xr - xl + s.weight)}
          height={f1(h)}
          rx={f1((s.cornerRadius ?? 0) + s.weight / 2)}
          fill={backing}
        />
      ) : null}
      {bracket(s.brackets, -1, xl, h, s.weight, ink, "l", s.cornerRadius ?? 0, s.roundEnds, arm)}
      {bracket(s.brackets, 1, xr, h, s.weight, ink, "r", s.cornerRadius ?? 0, s.roundEnds, arm)}
    </>
  );
}

/**
 * Stays on device: the face turns into an iPhone (a plain outline, no island), then a Mac. The brackets move to the device's
 * width and height and their arms grow until they meet in the middle, closing the outline, while
 * the eyes shrink onto the screen. The ears stay on: same size as on the face, sitting at the
 * device's top corners, pulled in only if a narrow phone needs the room. Corners stay as sharp
 * as the face's own corner radius setting.
 */
const PHONE = { half: 30, height: 100, eyes: 0.6 };
const MAC = { half: 60, height: 70, eyes: 0.85 };
function withDevice(s: Settings, f: Face) {
  const phone = f.phone ?? 0;
  const mac = f.mac ?? 0;
  const closing = Math.min(1, phone + mac);
  if (closing < 0.001) return { s, phone: 0, mac: 0, arm: undefined, devY: 0 };
  const ease = (x: number) => x * x * (3 - 2 * x);
  const shape = ease(closing);
  // Which device we're heading to (or between, during phone to Mac), scaled by how far along.
  const phoneW = (phone / closing) * shape;
  const macW = (mac / closing) * shape;
  const spread = s.spread + (PHONE.half - s.spread) * phoneW + (MAC.half - s.spread) * macW;
  const bracketHeight =
    s.bracketHeight + (PHONE.height - s.bracketHeight) * phoneW + (MAC.height - s.bracketHeight) * macW;
  const scale = 1 - (1 - PHONE.eyes) * phoneW - (1 - MAC.eyes) * macW;
  // Ears keep the face's size; on a device narrower than the ears' span they move in together.
  const ear = earSize(s);
  const span = ear.inset + ear.B;
  const room = (half: number) => Math.min(1, (half - Math.max(s.weight * 1.2, 4)) / Math.max(1, span));
  const fit = 1 - (1 - room(PHONE.half)) * phoneW - (1 - room(MAC.half)) * macW;
  const out: Settings = {
    ...s,
    spread,
    bracketHeight,
    eyeSize: s.eyeSize * scale,
    gap: s.gap * scale,
    earB: ear.B * fit,
    earInset: ear.inset * fit,
    armGrow: shape,
  };
  const arm = lerp(Math.max(s.weight * 1.6, bracketHeight * 0.2), spread, shape);
  return { s: out, phone: phoneW, mac: macW, arm, devY: -5 * macW };
}

/* ------------------------------------------------------------------ eye illustrations */

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const hash = (n: number) => {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
};

/**
 * Writing: the eyes become one page of text spanning the whole face, bracket to bracket. Tokens
 * land whole, a block cursor follows the last one, a paragraph's last line runs short, and the
 * page scrolls a line at a time.
 */
function lineTokens(j: number, W: number, gap: number, small = false) {
  const short = j % 3 === 2; // a paragraph ends every third line
  const n = small
    ? short
      ? 1
      : 2 + (hash(j) > 0.5 ? 1 : 0)
    : short
      ? 2 + (hash(j) > 0.5 ? 1 : 0)
      : 3 + Math.floor(hash(j) * 3);
  const raw = Array.from({ length: n }, (_, k) => 0.55 + hash(j * 7 + k + 1));
  const sum = raw.reduce((a, b) => a + b, 0);
  const room = W * (short ? 0.5 : 1) - gap * (n - 1);
  return raw.map((r) => (r / sum) * room);
}

function textBlock(
  x0: number,
  x1: number,
  cy: number,
  E: number,
  written: number,
  ink: string,
  small = false,
  round = 0,
): ReactNode {
  // Small sizes: two thick lines instead of three, fewer and longer tokens.
  const rows = small ? 2 : 3;
  const W = x1 - x0;
  const gap = Math.max(2, E * (small ? 0.12 : 0.08));
  const th = E * (small ? 0.3 : 0.17);
  const pitch = small ? E - th : E * 0.33;
  // Find the line being written and how many of its tokens are down.
  let left = Math.floor(written);
  let L = 0;
  for (;;) {
    const n = lineTokens(L, W, gap, small).length;
    if (left < n) break;
    left -= n;
    L += 1;
  }
  const first = Math.max(0, L - (rows - 1));
  const bars: ReactNode[] = [];
  for (let l = first; l <= L; l += 1) {
    const widths = lineTokens(l, W, gap, small);
    const shown = l < L ? widths.length : left;
    const y = cy - E / 2 + (small ? 0 : E * 0.085) + (l - first) * pitch;
    let x = x0;
    for (let k = 0; k < shown; k += 1) {
      bars.push(
        <rect
          key={`${l}-${k}`}
          x={f1(x)}
          y={f1(y)}
          width={f1(widths[k]!)}
          height={f1(th)}
          rx={f1((th / 2) * round)}
          fill={ink}
        />,
      );
      x += widths[k]! + gap;
    }
    if (l === L)
      bars.push(
        <rect
          key="cursor"
          x={f1(x)}
          y={f1(y)}
          width={f1(th * 0.9)}
          height={f1(th)}
          rx={f1((th / 2) * round)}
          fill={ink}
        />,
      );
  }
  return bars;
}

/* ------------------------------------------------------------------ pixel zeros */

/** Pixel zeros, as in pixel monospace fonts. Which inner mark each pixel symbol carries. */
const PIXEL_KINDS: Record<string, "none" | "slash" | "dot"> = {
  pixel: "none",
  "pixel-slashed": "slash",
  "pixel-dotted": "dot",
};

/**
 * A bitmap zero on a cols x rows grid: a top bar and a bottom bar (centred, their own widths, so
 * narrower bars cut the corners like a pixel font), sides of the given thickness, and an
 * optional slash (bottom-left to top-right) or centre dot inside.
 */
function pixelZero(s: Settings, inner: "none" | "slash" | "dot"): string[] {
  const cols = Math.round(Math.min(9, Math.max(3, s.pxCols ?? 5)));
  const rows = Math.round(Math.min(11, Math.max(4, s.pxRows ?? 7)));
  const side = Math.round(Math.min(Math.floor((cols - 1) / 2), Math.max(1, s.pxSide ?? 1)));
  const topW = Math.round(Math.min(cols, Math.max(1, s.pxTop ?? cols - 2)));
  const botW = Math.round(Math.min(cols, Math.max(1, s.pxBottom ?? cols - 2)));
  const g = Array.from({ length: rows }, () => Array.from({ length: cols }, () => "."));
  const run = (r: number, w: number) => {
    const from = Math.floor((cols - w) / 2);
    for (let c = from; c < from + w; c += 1) g[r]![c] = "#";
  };
  for (let r = 0; r < side; r += 1) run(r, topW);
  for (let r = rows - side; r < rows; r += 1) run(r, botW);
  for (let r = side; r < rows - side; r += 1) {
    for (let k = 0; k < side; k += 1) {
      g[r]![k] = "#";
      g[r]![cols - 1 - k] = "#";
    }
  }
  const first = side;
  const last = rows - side - 1;
  const left = side;
  const right = cols - side - 1;
  if (inner === "slash" && right >= left && last > first) {
    // Leave the interior rows next to the bars clear when there's room, as fonts do.
    const a = last - first >= 4 ? first + 1 : first;
    const b = last - first >= 4 ? last - 1 : last;
    for (let r = a; r <= b; r += 1) {
      const u = b === a ? 0.5 : (b - r) / (b - a);
      g[r]![left + Math.round(u * (right - left))] = "#";
    }
  }
  if (inner === "dot" && right >= left && last >= first) {
    const rs =
      (last - first) % 2 === 0 ? [(first + last) / 2] : [Math.floor((first + last) / 2), Math.ceil((first + last) / 2)];
    const cs =
      (right - left) % 2 === 0 ? [(left + right) / 2] : [Math.floor((left + right) / 2), Math.ceil((left + right) / 2)];
    rs.forEach((r) => cs.forEach((cc) => (g[r]![cc] = "#")));
  }
  return g.map((row) => row.join(""));
}

/**
 * One filled path for a glyph. With no gap, each row's runs merge so neighbouring pixels never
 * show seams; with a gap, every pixel is its own square, like an LED matrix.
 */
function pixelPath(rows: string[], x: number, y: number, cell: number, gap = 0, round = 0) {
  let d = "";
  const inset = (cell * gap) / 2;
  // Rounded pixels: every pixel is its own rounded square (radius as a share of the pixel).
  const r = Math.max(0, Math.min(0.5, round)) * (cell - 2 * inset);
  const separate = gap > 0 || r > 0.01;
  rows.forEach((row, ri) => {
    let c = 0;
    while (c < row.length) {
      if (row[c] !== "#") {
        c += 1;
        continue;
      }
      let end = c + 1;
      if (!separate) while (end < row.length && row[end] === "#") end += 1;
      const x0 = x + c * cell + inset;
      const y0 = y + ri * cell + inset;
      const x1 = x + end * cell - inset;
      const y1 = y + (ri + 1) * cell - inset;
      d +=
        r > 0.01
          ? `M${f1(x0 + r)} ${f1(y0)}H${f1(x1 - r)}A${f1(r)} ${f1(r)} 0 0 1 ${f1(x1)} ${f1(y0 + r)}V${f1(y1 - r)}A${f1(r)} ${f1(r)} 0 0 1 ${f1(x1 - r)} ${f1(y1)}H${f1(x0 + r)}A${f1(r)} ${f1(r)} 0 0 1 ${f1(x0)} ${f1(y1 - r)}V${f1(y0 + r)}A${f1(r)} ${f1(r)} 0 0 1 ${f1(x0 + r)} ${f1(y0)}Z`
          : `M${f1(x0)} ${f1(y0)}H${f1(x1)}V${f1(y1)}H${f1(x0)}Z`;
      c = end;
    }
  });
  return d;
}

/* ------------------------------------------------------------------ 1. matrix face */

/*
 * Everything here is a rectangle. Expressions are lids: the top lid travels most of a blink, the
 * bottom one rises to make the happy squint. Loading turns the eyes into outlines that fill.
 */
const matrixFace: Character = {
  id: "matrix",
  name: "Matrix face",
  draw({ f, s: base, ink, paper, uid = "stage", backing }) {
    const device = withDevice(base, f);
    const s = device.s;
    const spread = s.spread;
    const aspect = s.eyeAspect ?? 1;
    const restW = s.eyeSize * aspect;
    const small = s.detail === "medium" || s.detail === "tiny";
    const tiny = s.detail === "tiny";
    // One line weight for everything drawn as a line: brackets, ears, glasses, loader outlines.
    const line = s.weight;
    const cat = s.ears !== "none" && s.brackets === "square";
    // Ears add height on top, so the whole face drops to sit optically centred in its box.
    const shift = cat ? earSize(s).rise * 0.42 : 0;
    // Reading goggles: the eyes themselves become the lenses. Each eye hollows out into an
    // outline a touch bigger than the eye, a solid pupil inside scans the lines, a bridge joins
    // the lenses and straps run out to the brackets on both sides. The lenses hold still while
    // reading; only the pupils move. Gaps stay at least about a pixel at the size being drawn.
    const G = f.glasses;
    const grow = 1 - (1 - G) ** 3;
    const unit = s.unit ?? 0.4; // box units per pixel
    const clear = Math.max(2.5, unit * 1.1);
    const E = s.eyeSize * f.size * (1 + 0.1 * grow);
    // Lenses keep the eye's own shape (tall, with its corner rounding) but get wider than the eye,
    // up to ~0.72 of its height, so a pupil fits; the eyes move out to the lens centres to make
    // that room. The pupil is a small copy of the eye, same proportions, so it still reads as uzu.
    const innerHalf = spread - line / 2 - clear - line * 0.5;
    const bridgeMin = Math.max(line * 1.2, 4, unit * 1.5);
    const lensW = Math.max(E * aspect, Math.min(E * 0.72, innerHalf - bridgeMin / 2));
    const lensC = Math.min(Math.max(s.gap / 2, bridgeMin / 2 + lensW / 2), innerHalf - lensW / 2);
    const W = lerp(E * aspect, lensW, grow);
    const openX = W - 2 * line - 2 * clear;
    const openY = E - 2 * line - 2 * clear;
    const pupilH = Math.max(0, Math.min(openY * 0.74, (openX * 0.88) / Math.max(0.2, aspect)));
    const pupilW = pupilH * aspect;
    const showPupil = pupilW > unit * 0.8;
    const pupilX = Math.max(0, (openX - pupilW) / 2);
    const pupilY = Math.max(0, (openY - pupilH) / 2);
    const ox = Math.max(1.5, spread - s.weight - restW / 2 - s.gap / 2 - 3) * (1 - grow);
    const oy = Math.max(1.5, s.bracketHeight / 2 - s.weight - s.eyeSize / 2 - 3) * (1 - grow);
    const eyeC = lerp(s.gap / 2, lensC, grow);
    const minH = Math.max(2, s.weight * 0.55);
    const eyes = [-1, 1].map((side) => {
      const b = side < 0 ? f.blinkL : f.blinkR;
      const closed = Math.max(b, f.sleep);
      const squint = f.squint + (side === 1 ? 0.18 * f.asym : 0);
      // A hint of head turn: the eye on the side being looked at gets a little wider.
      const w = W * (1 + 0.1 * f.turn * side) * (1 + 0.05 * f.happy) * (1 + 0.03 * b);
      const cx = 80 + side * eyeC + f.gx * ox + f.jitter;
      const cy = 80 + f.gy * oy + f.lift - E * 0.05 * f.happy;
      const h = E;
      const bar = s.happyShape === "bar" ? f.happy : 0;
      let top = cy - h / 2 + h * (0.8 * closed + 0.2 * squint);
      let bottom = cy + h / 2 - h * (0.12 * closed + 0.15 * squint + 0.62 * bar);
      if (bottom - top < minH) {
        const mid = (top + bottom) / 2;
        top = mid - minH / 2;
        bottom = mid + minH / 2;
      }
      // Separate eyelids: the lid comes down from the top as its own line (a gap under it), resting
      // at the lid size and closing all the way for blinks and sleep. Only the top lid moves.
      let lid: { y: number; top: number; bottom: number } | null = null;
      if (s.eyelids) {
        const rest = Math.min(0.7, Math.max(0, s.lidSize ?? 0.3));
        const L = Math.min(1, rest + (1 - rest) * closed + 0.2 * squint);
        const lidY = cy - h / 2 + h * L;
        const gapUnder = Math.max(2, unit);
        lid = {
          y: lidY,
          top: lidY + line / 2 + gapUnder,
          bottom: cy + h / 2 - h * (0.15 * squint + 0.62 * bar),
        };
      }
      return {
        side,
        x: cx - w / 2,
        y: top,
        w,
        h: bottom - top,
        cx,
        cy,
        closed: Math.max(closed, b),
        lid,
      };
    });
    // Calling a tool: each eye turns round its own centre, and shrinks just enough at every angle
    // that its turned width stays inside the space between the eyes, so the two never touch.
    const gearTransform = (e: { side: number; w: number; h: number; cx: number; cy: number }) => {
      const spin = f.spin ?? 0;
      if (Math.abs(spin % 180) < 0.05) return undefined;
      const a = (spin * e.side * Math.PI) / 180;
      const turned = e.w * Math.abs(Math.cos(a)) + e.h * Math.abs(Math.sin(a));
      const room = s.gap - Math.max(3, line, unit * 1.2);
      const k = turned > room ? Math.max(0.2, room / turned) : 1;
      return `rotate(${f1(spin * e.side)} ${f1(e.cx)} ${f1(e.cy)}) translate(${f1(e.cx)} ${f1(e.cy)}) scale(${k.toFixed(3)}) translate(${f1(-e.cx)} ${f1(-e.cy)})`;
    };
    // Lids reach a little past the eye on both sides, but never so far that the two lids meet.
    const lidOver = Math.max(
      0,
      Math.min(line * 0.6, (eyes[1]!.x - (eyes[0]!.x + eyes[0]!.w) - Math.max(2, unit * 1.2)) / 2),
    );
    // The eye's symbol in a box: a solid block, or a zero (outlined, optionally slashed or dotted).
    // A zero squashed thinner than its own outline (mid-blink) is drawn solid, like a closed eye.
    const symbol = (
      box: { x: number; y: number; w: number; h: number },
      key: string | number,
      full?: { x: number; y: number; w: number; h: number },
    ) => {
      if (box.h < 0.3 || box.w < 0.3) return null;
      let kind = s.eyeSymbol ?? "block";
      if (kind === "none") return null; // icons draw their own symbol in place of the eyes
      // Heart eyes: a heart sized to the eye's height (and no wider than the space between the
      // eyes allows), centred on the eye. Half-closed (a blink) it reads as a closed eye.
      if (kind === "heart") {
        const area = full ?? box;
        if (box.h < area.h * 0.5)
          return (
            <rect key={key} x={f1(box.x)} y={f1(box.y)} width={f1(box.w)} height={f1(box.h)} rx={rx(box)} fill={ink} />
          );
        const z = Math.min(area.h * 1.05, s.gap - Math.max(3, line, unit * 1.2), Math.max(area.w, area.h) * 1.1);
        const hx = box.x + box.w / 2;
        const hy = area.y + area.h / 2 + z * 0.06;
        return (
          <path
            key={key}
            d={`M${f1(hx)} ${f1(hy + z * 0.42)} C${f1(hx - z * 0.62)} ${f1(hy + z * 0.02)} ${f1(hx - z * 0.5)} ${f1(hy - z * 0.52)} ${f1(hx)} ${f1(hy - z * 0.2)} C${f1(hx + z * 0.5)} ${f1(hy - z * 0.52)} ${f1(hx + z * 0.62)} ${f1(hy + z * 0.02)} ${f1(hx)} ${f1(hy + z * 0.42)} Z`}
            fill={ink}
          />
        );
      }
      // Pixel zeros: a 5x7 bitmap glyph with square pixels, laid out in the eye's full box and
      // cut by the lid (blink, eyelids) row by row rather than squashed. Too small to read (a
      // pixel under ~1.2 screen px) and they fall back to the outlined zero.
      const pk = PIXEL_KINDS[kind];
      if (pk) {
        const glyph = pixelZero(s, pk);
        const cols = glyph[0]!.length;
        const rowsN = glyph.length;
        const area = full ?? box;
        const cell = Math.min(area.w / cols, area.h / rowsN);
        if (cell >= unit * 1.2) {
          if (box.h < cell * 1.2)
            return <rect key={key} x={f1(box.x)} y={f1(box.y)} width={f1(box.w)} height={f1(box.h)} fill={ink} />;
          const gx0 = area.x + (area.w - cell * cols) / 2;
          const gy0 = area.y + (area.h - cell * rowsN) / 2;
          const id = `uzu-px-${uid}-${key}`;
          return (
            <g key={key}>
              <clipPath id={id}>
                <rect x={f1(box.x - 1)} y={f1(box.y)} width={f1(box.w + 2)} height={f1(box.h)} />
              </clipPath>
              <path
                d={pixelPath(glyph, gx0, gy0, cell, Math.min(0.45, Math.max(0, s.pxGap ?? 0)), s.pxRound ?? 0)}
                fill={ink}
                clipPath={`url(#${id})`}
              />
            </g>
          );
        }
        kind = "zero";
      }
      const solid = kind === "block" || box.h < 2 * line + 1.2 || box.w < 2 * line + 1.2;
      if (solid)
        return (
          <rect key={key} x={f1(box.x)} y={f1(box.y)} width={f1(box.w)} height={f1(box.h)} rx={rx(box)} fill={ink} />
        );
      const innerW = box.w - 2 * line;
      const innerH = box.h - 2 * line;
      const detail = tiny ? "zero" : kind;
      const dot = Math.min(innerW, innerH) * 0.42;
      return (
        <g key={key}>
          <rect
            x={f1(box.x + line / 2)}
            y={f1(box.y + line / 2)}
            width={f1(box.w - line)}
            height={f1(box.h - line)}
            rx={outlineRx(box)}
            fill="none"
            stroke={ink}
            strokeWidth={f1(line)}
          />
          {detail === "slashed" && innerW > line * 1.5 && innerH > line * 1.5 ? (
            <path
              d={`M${f1(box.x + line)} ${f1(box.y + box.h - line)} L${f1(box.x + box.w - line)} ${f1(box.y + line)}`}
              stroke={ink}
              strokeWidth={f1(line * 0.9)}
            />
          ) : null}
          {detail === "dotted" && dot > unit ? (
            <rect
              x={f1(box.x + box.w / 2 - dot / 2)}
              y={f1(box.y + box.h / 2 - dot / 2)}
              width={f1(dot)}
              height={f1(dot)}
              fill={ink}
            />
          ) : null}
        </g>
      );
    };
    // Happy as a chevron: the eye's six outline points slide to a ^ (or v) of the same width.
    // Both arms keep one perpendicular thickness, so the tip is a true miter, not a lump.
    const chevron = s.happyShape === "caret" || s.happyShape === "vee" ? f.happy : 0;
    // Rounded eyes stay rounded through the morph: the shape is drawn inset by half of a round
    // stroke `k` and then stroked with round joins and caps, which is exactly a rounded shape. At
    // the start k matches the eye's own corner radius (a pill stays a pill); at the end it is the
    // v's full thickness, so the happy eye is a soft v with round ends and a round tip.
    const eyePath = (e: { x: number; y: number; w: number; h: number; cx: number; cy: number }) => {
      const dir = s.happyShape === "vee" ? 1 : -1;
      const A = E * 0.5;
      const thick = Math.max(line * 1.6, E * 0.17, minH * 1.3);
      const cosA = Math.cos(Math.atan2(A, e.w / 2));
      const rRect = Math.min(Math.min(e.w, e.h) * s.eyeRadius, Math.min(e.w, e.h) / 2);
      const k = s.eyeRadius > 0 ? lerp(2 * rRect, Math.min(thick, 2 * rRect + thick * 0.5), chevron) : 0;
      const t = Math.max(0, thick / cosA - k / cosA);
      const armsY = e.cy - (dir * A) / 2;
      const apexY = e.cy + (dir * A) / 2;
      const x0 = e.x + k / 2;
      const x1 = e.x + e.w - k / 2;
      const y0 = e.y + k / 2;
      const y1 = e.y + e.h - k / 2;
      const from: Array<[number, number]> = [
        [x0, y0],
        [e.cx, y0],
        [x1, y0],
        [x1, y1],
        [e.cx, y1],
        [x0, y1],
      ];
      const to: Array<[number, number]> = [
        [x0, armsY - t / 2],
        [e.cx, apexY - t / 2],
        [x1, armsY - t / 2],
        [x1, armsY + t / 2],
        [e.cx, apexY + t / 2],
        [x0, armsY + t / 2],
      ];
      const pts = from.map(
        ([fx, fy], i) => [fx + (to[i]![0] - fx) * chevron, fy + (to[i]![1] - fy) * chevron] as [number, number],
      );
      return {
        d: `M${pts.map(([x, y]) => `${f1(x)} ${f1(y)}`).join(" L")} Z`,
        k,
      };
    };
    // Soft styles (pill eyes, round line ends, rounded brackets) round the small parts too:
    // lids, text lines, the MacBook base get round ends; the tool square and sleep squares follow
    // the eye's roundness.
    const soft = s.roundEnds === true || s.eyeRadius >= 0.25 || (s.cornerRadius ?? 0) >= 6 ? 1 : 0;
    const dotR = (size: number) => f1(size * Math.min(0.5, s.eyeRadius));
    const rx = (e: { w: number; h: number }) => f1(Math.min(e.w, e.h) * s.eyeRadius);
    // An outlined eye (loader, goggle lens): the stroke sits half a line inside the eye's box, so
    // its radius is the eye's radius less half a line, and the outside matches the solid eye.
    const outlineRx = (e: { w: number; h: number }) => f1(Math.max(0, Math.min(e.w, e.h) * s.eyeRadius - line / 2));
    const loading = f.fill >= 0;
    // Whole-face tremble, a little larger on small drawings so it still reads in pixels.
    const shake = f.shake * (tiny ? 6 : small ? 3 : 1);
    const bob = (f.bob ?? 0) * (tiny ? 3 : small ? 2 : 1);
    // Resting squares drift up from above the right ear (or the right bracket's corner).
    const zx = cat ? 80 + spread - earSize(s).inset - earSize(s).tip * earSize(s).B + line : 80 + spread - 4;
    const zy = 80 - s.bracketHeight / 2 - (cat ? earSize(s).rise + line : 2);
    const zScale = tiny ? 2.6 : small ? 1.7 : 1;
    return (
      <g
        transform={
          shift || shake || bob || device.devY || f.tilt
            ? `translate(${f1(shake)} ${f1(shift + device.devY + bob)}) rotate(${f1(f.tilt ?? 0)} 80 80)`
            : undefined
        }
      >
        <Brackets
          f={f}
          s={s}
          ink={ink}
          spread={spread}
          height={s.bracketHeight}
          left={f.bracketL}
          right={f.bracketR}
          arm={device.arm}
          backing={backing}
        />
        {device.mac > 0.6 ? (
          // The MacBook's base: a plain, wider bar under the screen.
          <g>
            <rect
              x={f1(80 - spread - 11 * ((device.mac - 0.6) / 0.4))}
              y={f1(80 + s.bracketHeight / 2 + 2.5)}
              width={f1(2 * spread + 22 * ((device.mac - 0.6) / 0.4))}
              height={f1(line * 1.1)}
              rx={f1(line * 0.55 * soft)}
              fill={ink}
            />
          </g>
        ) : null}
        {eyes.map((e) =>
          f.look !== "eyes" && e.closed < 0.6 ? (
            // One page across the whole face (drawn once, with the left eye), revealed by the lids.
            e.side < 0 ? (
              <g key={e.side}>
                <clipPath id={`uzu-lid-${uid}`}>
                  <rect x={f1(80 - spread)} y={f1(e.y)} width={f1(2 * spread)} height={f1(e.h)} />
                </clipPath>
                <g clipPath={`url(#uzu-lid-${uid})`}>
                  {textBlock(
                    80 - spread + line / 2 + Math.max(4, line, unit * 1.2),
                    80 + spread - line / 2 - Math.max(4, line, unit * 1.2),
                    e.cy,
                    E,
                    f.text,
                    ink,
                    small,
                    soft,
                  )}
                </g>
              </g>
            ) : null
          ) : G > 0.02 ? (
            // A goggle lens: the eye's own box as an outline, the eye shrinking into the pupil.
            <g key={e.side}>
              <rect
                x={f1(e.x + line / 2)}
                y={f1(e.y + line / 2)}
                width={f1(Math.max(0, e.w - line))}
                height={f1(Math.max(0, e.h - line))}
                rx={outlineRx(e)}
                fill={paper}
                stroke={ink}
                strokeWidth={f1(line)}
              />
              {(() => {
                const pw = lerp(e.w, showPupil ? pupilW : 0, grow);
                const ph = lerp(e.h, showPupil ? pupilH : 0, grow);
                if (pw < 0.3 || ph < 0.3) return null;
                const pcx = e.cx + f.gx * pupilX * grow;
                const pcy = e.cy + f.gy * pupilY * grow;
                return (
                  <rect
                    x={f1(pcx - pw / 2)}
                    y={f1(pcy - ph / 2)}
                    width={f1(pw)}
                    height={f1(ph)}
                    rx={rx({ w: pw, h: ph })}
                    fill={ink}
                  />
                );
              })()}
            </g>
          ) : loading ? (
            <g key={e.side}>
              {/* The fill rises inside the eye's own rounded shape: flat on top, corners below. */}
              <clipPath id={`uzu-load-${uid}-${e.side}`}>
                <rect x={f1(e.x)} y={f1(e.y)} width={f1(e.w)} height={f1(e.h)} rx={rx(e)} />
              </clipPath>
              <rect
                x={f1(e.x + line / 2)}
                y={f1(e.y + line / 2)}
                width={f1(Math.max(0, e.w - line))}
                height={f1(Math.max(0, e.h - line))}
                rx={outlineRx(e)}
                fill={paper}
                stroke={ink}
                strokeWidth={f1(line)}
              />
              <rect
                x={f1(e.x)}
                y={f1(e.y + e.h * (1 - f.fill))}
                width={f1(e.w)}
                height={f1(e.h * f.fill)}
                fill={ink}
                clipPath={`url(#uzu-load-${uid}-${e.side})`}
              />
            </g>
          ) : chevron > 0.001 ? (
            (() => {
              const shape = eyePath(e);
              return shape.k > 0.05 ? (
                <path
                  key={e.side}
                  d={shape.d}
                  fill={ink}
                  stroke={ink}
                  strokeWidth={f1(shape.k)}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
              ) : (
                <path key={e.side} d={shape.d} fill={ink} />
              );
            })()
          ) : (
            // Calling a tool: the eyes turn like two meshing gears, each round its own centre.
            <g key={e.side} transform={gearTransform(e)}>
              {e.lid
                ? symbol(
                    {
                      x: e.x,
                      y: e.lid.top,
                      w: e.w,
                      h: Math.max(0, e.lid.bottom - e.lid.top),
                    },
                    e.side,
                    { x: e.x, y: e.cy - E / 2, w: e.w, h: E },
                  )
                : symbol(e, e.side, { x: e.x, y: e.cy - E / 2, w: e.w, h: E })}
              {e.lid
                ? (() => {
                    // Over a pixel zero the lid spans the glyph, not the wider eye box.
                    const pc = Math.round(s.pxCols ?? 5);
                    const cell = Math.min(e.w / pc, E / Math.round(s.pxRows ?? 7));
                    const lw = PIXEL_KINDS[s.eyeSymbol ?? ""] && cell >= unit * 1.2 ? cell * pc : e.w;
                    return (
                      <rect
                        x={f1(e.cx - lw / 2 - lidOver)}
                        y={f1(e.lid.y - line / 2)}
                        width={f1(lw + lidOver * 2)}
                        height={f1(line)}
                        rx={f1((line / 2) * soft)}
                        fill={ink}
                      />
                    );
                  })()
                : null}
            </g>
          ),
        )}
        {G > 0.5
          ? (() => {
              // Bridge between the lenses and straps out to the brackets, growing in once the
              // lenses have formed.
              const k = Math.min(1, (G - 0.5) / 0.45);
              const [l, r] = [eyes[0]!, eyes[1]!];
              const y = (l.cy + r.cy) / 2 - line / 2;
              const bk = s.detail === "tiny" ? 1.8 : s.detail === "medium" ? 1.4 : 1;
              const spineL = 80 - spread - f.bracketL * bk + line / 2;
              const spineR = 80 + spread + f.bracketR * bk - line / 2;
              const gapIn = r.x - (l.x + l.w);
              const outL = l.x - spineL;
              const outR = spineR - (r.x + r.w);
              return (
                <g>
                  {gapIn > 0.2 ? (
                    <rect
                      x={f1((l.x + l.w + r.x) / 2 - (gapIn * k) / 2 - 0.3)}
                      y={f1(y)}
                      width={f1(gapIn * k + 0.6)}
                      height={f1(line)}
                      fill={ink}
                    />
                  ) : null}
                  {outL > 0.2 ? (
                    <rect
                      x={f1(l.x - outL * k - 0.3)}
                      y={f1(y)}
                      width={f1(outL * k + 0.6)}
                      height={f1(line)}
                      fill={ink}
                    />
                  ) : null}
                  {outR > 0.2 ? (
                    <rect x={f1(r.x + r.w - 0.3)} y={f1(y)} width={f1(outR * k + 0.6)} height={f1(line)} fill={ink} />
                  ) : null}
                </g>
              );
            })()
          : null}
        {f.zs.map((z) => {
          // Resting: small squares drift up and away, solid ink, growing in and shrinking out.
          const u = z.age / 3;
          const size = Math.max(0, 4.2 * zScale * Math.min(1, u * 6) * Math.min(1, (1 - u) * 4)) * (0.8 + 0.5 * u);
          if (size < 0.4) return null;
          const x = zx + 8 * u + Math.sin(z.age * 2) * 2;
          const y = zy - 30 * u;
          return (
            <rect
              key={z.id}
              x={f1(x - size / 2)}
              y={f1(y - size / 2)}
              width={f1(size)}
              height={f1(size)}
              rx={dotR(size)}
              fill={ink}
            />
          );
        })}
      </g>
    );
  },
};

export const drawFace = matrixFace.draw;
