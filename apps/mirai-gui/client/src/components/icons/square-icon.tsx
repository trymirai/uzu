import type { SVGProps } from "react";

export type SquareIconProps = SVGProps<SVGSVGElement> & {
  size?: number;
  color?: string;
};

export function SquareIcon({ size = 16, color = "currentColor", ...props }: SquareIconProps) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill={color} {...props}>
      <rect x="3" y="3" width="18" height="18" rx="3" />
    </svg>
  );
}
