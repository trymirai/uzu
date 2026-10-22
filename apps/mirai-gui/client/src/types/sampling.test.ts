import { describe, expect, it } from "vitest";
import { normalizeSampling } from "./sampling";

describe("normalizeSampling", () => {
  it("turns a saved zero temperature into greedy", () => {
    expect(normalizeSampling({ type: "Stochastic", temperature: 0, topK: 40 })).toEqual({ type: "Greedy" });
  });

  it("migrates the old Argmax name without mistaking disabled temperature scaling for greedy", () => {
    expect(normalizeSampling({ type: "Argmax" })).toEqual({ type: "Greedy" });
    expect(normalizeSampling({ type: "Stochastic", temperature: null, topK: null })).toEqual({
      type: "Stochastic",
      temperature: null,
      topK: null,
    });
  });

  it("keeps everything else as saved", () => {
    const stochastic = { type: "Stochastic", temperature: 0.7, topK: 40 } as const;
    expect(normalizeSampling(stochastic)).toBe(stochastic);
    expect(normalizeSampling({ type: "Default" })).toEqual({ type: "Default" });
  });
});
