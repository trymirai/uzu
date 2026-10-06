import type { SVGProps } from "react";

export type IconCheckCircleFilledProps = SVGProps<SVGSVGElement> & {
  size?: number;
  color?: string;
};

export function IconCheckCircleFilled({ size = 16, color = "currentColor", ...props }: IconCheckCircleFilledProps) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 16 16" fill="none" {...props}>
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M7.99967 1.33337C11.6815 1.33337 14.6663 4.31814 14.6663 8.00004C14.6663 11.6819 11.6815 14.6667 7.99967 14.6667C4.31777 14.6667 1.33301 11.6819 1.33301 8.00004C1.33301 4.31814 4.31777 1.33337 7.99967 1.33337ZM10.3721 5.44731C10.0667 5.24177 9.65261 5.32241 9.44694 5.62765L6.98147 9.28911L5.80436 8.11204C5.54401 7.85164 5.122 7.85164 4.86165 8.11204C4.6013 8.37237 4.6013 8.79437 4.86165 9.05471L6.61165 10.8047C6.75254 10.9456 6.94921 11.016 7.14747 10.9968C7.34554 10.9775 7.52454 10.8708 7.63574 10.7058L10.5524 6.37243C10.7579 6.06707 10.6773 5.65295 10.3721 5.44731Z"
        fill={color}
      />
    </svg>
  );
}
