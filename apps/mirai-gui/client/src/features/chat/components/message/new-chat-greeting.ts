/*
 * What uzu says above an empty new chat. The first new chat ever gets the plain introduction; after
 * that it's a different line each time, never the same one twice in a row, the way a friend doesn't
 * say hello the same way every time. Lines stay short, in uzu's voice, and promise nothing beyond
 * what the introduction does: the chat stays on this device.
 */

export type Greeting = { title: string; subtitle: string };

export const FIRST_GREETING: Greeting = { title: "Start a private chat", subtitle: "It stays fully on your device" };

export const GREETINGS: readonly Greeting[] = [
  { title: "What's on your mind?", subtitle: "I'm all ears, and they stay right here" },
  { title: "Purr-fectly private", subtitle: "Everything we say stays on this device" },
  { title: "Back for more?", subtitle: "I kept your spot warm" },
  { title: "Ready when you are", subtitle: "Just you, me, and your device" },
  { title: "I was napping", subtitle: "But I'm up now. What's new?" },
  { title: "Ask me anything", subtitle: "I'll think it over right here" },
  { title: "Let's figure it out", subtitle: "Two heads, one device" },
  { title: "Psst, it's just us", subtitle: "No cloud listening in" },
  { title: "Fresh page, fresh paws", subtitle: "Where should we start?" },
  { title: "Curious about something?", subtitle: "So am I. Let's look into it" },
  { title: "Tell me everything", subtitle: "It won't leave this device" },
  { title: "Let's think it through", subtitle: "Slowly, privately, together" },
];

const SEEN_KEY = "mirai.greeting-seen";
const LAST_KEY = "mirai.greeting-last";

/** -1 is the first-time introduction; otherwise an index into GREETINGS. */
export type GreetingChoice = { index: number; greeting: Greeting };

/**
 * Chooses without writing anything, so it can run during render (and twice, under React's strict
 * mode) without spending the first-time introduction. `rememberGreeting` records what was shown.
 */
export function chooseGreeting(random: () => number = Math.random): GreetingChoice {
  try {
    if (localStorage.getItem(SEEN_KEY) !== "1") return { index: -1, greeting: FIRST_GREETING };
    const last = Number(localStorage.getItem(LAST_KEY) ?? -1);
    let index = Math.floor(random() * GREETINGS.length);
    if (index === last) index = (index + 1) % GREETINGS.length;
    return { index, greeting: GREETINGS[index] ?? FIRST_GREETING };
  } catch {
    // Storage unavailable: the introduction is always a fine thing to say.
    return { index: -1, greeting: FIRST_GREETING };
  }
}

export function rememberGreeting(choice: GreetingChoice): void {
  try {
    localStorage.setItem(SEEN_KEY, "1");
    if (choice.index >= 0) localStorage.setItem(LAST_KEY, String(choice.index));
  } catch {
    // Nothing to keep; the next new chat simply chooses again.
  }
}
