type TranscriptMetadata = { start: number; end: number; json: string };

// Content is opaque, including examples of our own file markers. A transcript
// belongs to the message/version header, before its reasoning or response.
export function* transcriptMetadata(markdown: string): Generator<TranscriptMetadata, undefined> {
  const markers =
    /^(?:(## (?:👤|🤖) .+ - .+|#### Version \d+.*)|<!-- (?:START_(CONTENT|COT|ERROR|PERF)|END_(CONTENT|COT|ERROR|PERF)|TRANSCRIPT: (.+)) -->)$/gm;
  let inHeader = false;
  let body: string | undefined;
  for (const match of markdown.matchAll(markers)) {
    const [, heading, start, end, json] = match;
    if (body) {
      if (end === body) body = undefined;
    } else if (heading) {
      inHeader = true;
    } else if (start) {
      body = start;
      if (start === "CONTENT" || start === "COT") inHeader = false;
    } else if (inHeader && json !== undefined) {
      yield { start: match.index, end: match.index + match[0].length, json };
    }
  }
}
