import { useState } from "react";
import { twMerge } from "tailwind-merge";
import type { VendorIconProps } from "./types";

export function VendorIcon({ src, vendor, className, size = 16 }: VendorIconProps) {
  const [failedSrc, setFailedSrc] = useState<string | null>(null);
  if (!src || failedSrc === src) return null;
  return (
    <img
      src={src}
      alt={vendor}
      width={size}
      height={size}
      draggable={false}
      onError={() => setFailedSrc(src)}
      className={twMerge("shrink-0 object-contain", className)}
    />
  );
}
