import { type Face, REST } from "./engine";

/** Still faces for icons, avatars and patterns: one frame, no engine. Gaze values are final. */
export type PoseId =
  "default" | "happy" | "wink" | "thinking" | "sleepy" | "goggles" | "loading" | "listening" | "lookLeft" | "lookRight";

export const POSES: Record<PoseId, { label: string; face: Partial<Face> }> = {
  default: { label: "Default", face: {} },
  happy: { label: "Happy", face: { happy: 1 } },
  wink: { label: "Wink", face: { blinkR: 1 } },
  thinking: { label: "Thinking", face: { gx: 0.35, gy: -0.65, squint: 0.3, asym: 1, turn: 0.3 } },
  sleepy: { label: "Asleep", face: { sleep: 1 } },
  goggles: { label: "Goggles", face: { glasses: 1 } },
  loading: { label: "Loading", face: { fill: 0.6 } },
  listening: { label: "Listening", face: { size: 1.08, earH: 1.15, earL: -6, earR: -6 } },
  lookLeft: { label: "Looking left", face: { gx: -0.6, turn: -0.5 } },
  lookRight: { label: "Looking right", face: { gx: 0.6, turn: 0.5 } },
};

export function poseFace(id: PoseId, extra: Partial<Face> = {}): Face {
  return { ...REST, ...POSES[id].face, ...extra };
}
