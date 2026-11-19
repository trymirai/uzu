import { describe, expect, it } from "vitest";
import { choosePettingLine, PETTING_LINES } from "./pet-cat";

describe("choosePettingLine", () => {
  it("picks one of the lines", () => {
    expect(PETTING_LINES[choosePettingLine(-1, () => 0)]).toBe("Prrr… that's the spot");
    expect(choosePettingLine(-1, () => 0.999)).toBe(PETTING_LINES.length - 1);
  });

  it("never says the same line twice in a row", () => {
    expect(choosePettingLine(0, () => 0)).toBe(1);
    expect(choosePettingLine(PETTING_LINES.length - 1, () => 0.999)).toBe(0);
  });
});
