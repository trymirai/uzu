import type { SVGProps } from "react";

export type IconPlayFilledProps = SVGProps<SVGSVGElement> & { size?: number; color?: string };

export function IconPlayFilled({ size = 16, color = "currentColor", ...props }: IconPlayFilledProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path
        d="M5.87355 2.05613C4.6186 1.28101 3 2.18373 3 3.65877V12.3413C3 13.8163 4.6186 14.7191 5.87355 13.9439L12.9022 9.60265C14.094 8.86657 14.094 7.13345 12.9022 6.39739L5.87355 2.05613Z"
        fill={color}
      />
    </svg>
  );
}
