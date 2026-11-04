import { twMerge } from "tailwind-merge";

// Catalog names carry the quantization as a long suffix ("Qwen3.5 2B Mirai Medium Quantization").
// The model is the first part; the variant is shortened to "Mirai M" and set quietly after it.
const VARIANT_SUFFIX = /^(.+?)\s+Mirai\s+(\S+)(?:\s+Quantization)?$/i;

export function splitModelName(name: string): { base: string; variant: string | null } {
  const match = VARIANT_SUFFIX.exec(name.trim());
  return match?.[1] && match[2]
    ? { base: match[1], variant: `Mirai ${match[2].charAt(0).toUpperCase()}` }
    : { base: name, variant: null };
}

/** A model's name with its quantization, shortened, as muted text after it. */
export function ModelName({ name, className }: { name: string; className?: string }) {
  const { base, variant } = splitModelName(name);
  return (
    <span className={twMerge("flex min-w-0 items-baseline gap-1.5", className)} title={name}>
      <span className="min-w-0 truncate">{base}</span>
      {variant ? <span className="shrink-0 whitespace-nowrap text-text-muted">{variant}</span> : null}
    </span>
  );
}
