import type { SVGProps } from "react";

export type IconSendProps = SVGProps<SVGSVGElement> & { size?: number; color?: string };

export function IconSend({ size = 16, color = "currentColor", ...props }: IconSendProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path d="M8 3V12" stroke={color} strokeWidth="1.33333" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M4 7L8 3L12 7" stroke={color} strokeWidth="1.33333" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
