import type { SVGProps } from "react";

export type IconPauseFilledProps = SVGProps<SVGSVGElement> & { size?: number; color?: string };

export function IconPauseFilled({ size = 16, color = "currentColor", ...props }: IconPauseFilledProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path
        d="M4.25 3C4.11193 3 4 3.12437 4 3.27778V12.7222C4 12.8756 4.11193 13 4.25 13H6.75C6.88807 13 7 12.8756 7 12.7222V3.27778C7 3.12437 6.88807 3 6.75 3H4.25Z"
        fill={color}
      />
      <path
        d="M9.25 3C9.11195 3 9 3.12437 9 3.27778V12.7222C9 12.8756 9.11195 13 9.25 13H11.75C11.8881 13 12 12.8756 12 12.7222V3.27778C12 3.12437 11.8881 3 11.75 3H9.25Z"
        fill={color}
      />
    </svg>
  );
}
