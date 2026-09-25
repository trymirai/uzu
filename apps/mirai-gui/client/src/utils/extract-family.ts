const stripSize = (s: string): string => s.replace(/-\d+(?:\.\d+)?B$/i, "");
const stripVariant = (s: string): string => s.replace(/-(?:Instruct|AWQ)$/i, "");

const applyRulesOnce = (s: string): string => [stripVariant, stripSize].reduce((acc, fn) => fn(acc), s);

export function extractFamily(name: string): string {
  const base = String(name || "").trim();
  const next = applyRulesOnce(base);
  return next === base ? base : extractFamily(next);
}
