import { expect, it } from "vitest";
import type { TranscriptItem } from "@/types/llm-stream";
import { collectSources } from "@/utils/web-sources";

const sources = [
  { title: "Rust Versions", url: "https://www.releases.rs/" },
  { title: "Rust Release Notes", url: "https://doc.rust-lang.org/stable/releases.html" },
];

it("collects every found or read page of a reply once, in the order the model saw them", () => {
  const extra = { title: "Rust Blog", url: "https://blog.rust-lang.org/" };
  const read = { title: "Read page", url: "https://read.example/" };
  const transcript: TranscriptItem[] = [
    { type: "thinking", text: "Look it up", completed: true },
    { type: "webSearch", query: "rust stable", called: true, sources },
    { type: "toolCall", name: "get_current_date_time", called: true },
    { type: "webSearch", query: "rust beta", called: true, sources: [sources[1]!, extra] },
    { type: "webSearch", query: "failed", called: true, failed: true },
    { type: "webPage", url: "https://blog.rust-lang.org/", called: true, source: extra },
    { type: "webPage", url: "https://read.example/", called: true, source: read },
    { type: "webPage", url: "https://refused.example/", called: true, failed: true },
    { type: "text", text: "Answer" },
  ];
  expect(collectSources(transcript)).toEqual([...sources, extra, read]);
});

it("lists a page once however a search and a read spelled its address", () => {
  const found = { title: "Example", url: "http://Example.com/x#top" };
  const read = { title: "Example page", url: "https://example.com/x/" };
  const transcript: TranscriptItem[] = [
    { type: "webSearch", query: "example", called: true, sources: [found] },
    { type: "webPage", url: "https://example.com/x/", called: true, source: read },
    {
      type: "webPage",
      url: "https://example.com/x?v=2",
      called: true,
      source: { ...read, url: "https://example.com/x?v=2" },
    },
  ];
  expect(collectSources(transcript)).toEqual([found, { ...read, url: "https://example.com/x?v=2" }]);
});
