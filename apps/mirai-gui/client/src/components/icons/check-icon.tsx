import type { SVGProps } from "react";

export type CheckIconProps = SVGProps<SVGSVGElement> & { size?: number; color?: string };

export function CheckIcon({ size = 16, color = "currentColor", ...props }: CheckIconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path d="M3 9L6 12L13 4" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
