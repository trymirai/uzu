import type { InputHTMLAttributes, ReactNode } from "react";

export type InputSize = "sm" | "md";

export type InputKind = "default" | "error";

export type InputProps = Omit<InputHTMLAttributes<HTMLInputElement>, "size" | "className"> & {
  size?: InputSize;
  kind?: InputKind;
  leftIcon?: ReactNode;
  fullWidth?: boolean;

  className?: string;
};
