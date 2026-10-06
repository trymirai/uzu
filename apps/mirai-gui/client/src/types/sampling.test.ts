import { describe, expect, it } from "vitest";
import { normalizeSampling } from "./sampling";

describe("normalizeSampling", () => {
  it("turns a saved zero temperature into argmax", () => {
    expect(normalizeSampling({ type: "Stochastic", temperature: 0, topK: 40 })).toEqual({ type: "Argmax" });
  });

  it("keeps everything else as saved", () => {
    const stochastic = { type: "Stochastic", temperature: 0.7, topK: 40 } as const;
    expect(normalizeSampling(stochastic)).toBe(stochastic);
    expect(normalizeSampling({ type: "Default" })).toEqual({ type: "Default" });
  });
});
