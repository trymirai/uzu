import type { CSSProperties } from "react";

type SpinnerProps = {
  size?: number;
  className?: string;
};

export function Spinner({ size = 16, className = "" }: SpinnerProps) {
  const style: CSSProperties = {
    width: size,
    height: size,
    animationDuration: "0.8s",
  };
  const resolvedClassName = className ? `animate-spin ${className}` : "animate-spin";

  return (
    <svg
      className={resolvedClassName}
      style={style}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" opacity="0.25" />
      <path d="M12 2C6.48 2 2 6.48 2 12" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}
