export type SelectorSize = "sm" | "md";

export type CheckboxProps = {
  checked: boolean;
  onChange: (checked: boolean) => void;
  size?: SelectorSize;
  disabled?: boolean;
  className?: string;
  "aria-label"?: string;
};
