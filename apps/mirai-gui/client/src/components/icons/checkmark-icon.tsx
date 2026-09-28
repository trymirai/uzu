import type { SVGProps } from "react";

export type IconCheckmarkProps = SVGProps<SVGSVGElement> & {
  size?: number;
  color?: string;
};

export function IconCheckmark({ size = 16, color = "currentColor", strokeWidth, ...props }: IconCheckmarkProps) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 20 20" fill="none" {...props}>
      <path
        d="M5 10l3.5 3.5L15 7"
        stroke={color}
        strokeWidth={strokeWidth ?? 1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
