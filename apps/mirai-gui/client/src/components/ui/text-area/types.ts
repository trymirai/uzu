import type { ComponentPropsWithoutRef } from "react";

export type TextAreaProps = Omit<ComponentPropsWithoutRef<"textarea">, "value" | "onChange" | "className"> & {
  value: string | null;
  onChange: (value: string) => void;
  maxHeightPx?: number;
  className?: string;
};
