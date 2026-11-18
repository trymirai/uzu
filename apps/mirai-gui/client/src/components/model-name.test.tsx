import { describe, expect, it } from "vitest";
import { splitModelName } from "./model-name";

describe("splitModelName", () => {
  it("shortens the quantization the way the website writes it", () => {
    expect(splitModelName("Qwen3.5 2B Mirai Medium Quantization")).toEqual({ base: "Qwen3.5 2B", variant: "Mirai M" });
    expect(splitModelName("Qwen3.8 27B Mirai Large")).toEqual({ base: "Qwen3.8 27B", variant: "Mirai L" });
    expect(splitModelName("Qwen3.8 27B Mirai Extra-Small Quantization")).toEqual({
      base: "Qwen3.8 27B",
      variant: "Mirai XS",
    });
    expect(splitModelName("Qwen3.8 27B Mirai Extra-Large")).toEqual({ base: "Qwen3.8 27B", variant: "Mirai XL" });
  });

  it("leaves names without a quantization alone", () => {
    expect(splitModelName("Llama 3.2 3B Instruct")).toEqual({ base: "Llama 3.2 3B Instruct", variant: null });
  });
});
