import { expect, it } from "vitest";
import { Roles } from "@/types/chat";
import type { Message } from "@/types/message";
import { buildChatRunInput } from "./build-chat-run-input";

const at = 1_700_000_000_000;

it("sends the pages an earlier answer found or read along with its text", () => {
  const history: Message[] = [
    { id: "u1", sender: Roles.User, text: "What is new in Rust?", timestamp: at },
    {
      id: "a1",
      sender: Roles.Assistant,
      text: "Rust 1.99 is out.",
      timestamp: at + 1,
      output: {
        transcript: [
          {
            type: "webSearch",
            query: "rust release",
            called: true,
            sources: [
              { title: "Releases", url: "https://releases.rs/" },
              { title: "Blog", url: "https://blog.rust-lang.org/" },
            ],
          },
          { type: "webPage", url: "", called: false },
          { type: "webPage", url: "https://made-up.example/secret?q=1", called: true, failed: true },
          {
            type: "webPage",
            url: "https://blog.rust-lang.org/",
            called: true,
            source: { title: "Blog", url: "https://blog.rust-lang.org/2026/01/01/Rust-1.99.0/" },
          },
          { type: "text", text: "Rust 1.99 is out." },
        ],
      },
    },
  ];

  const { messages } = buildChatRunInput({ history, prompt: "Open the second one", attachments: [] });

  expect(messages).toEqual([
    { role: Roles.User, content: "What is new in Rust?" },
    {
      role: Roles.Assistant,
      content: "Rust 1.99 is out.",
      openableUrls: [
        "https://releases.rs/",
        "https://blog.rust-lang.org/",
        "https://blog.rust-lang.org/2026/01/01/Rust-1.99.0/",
      ],
    },
    { role: Roles.User, content: "Open the second one" },
  ]);
});
