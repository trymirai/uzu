import { forwardRef, type ElementType } from "react";
import { twMerge } from "tailwind-merge";
import { SELECT_ROW_ACTIVE_CLASSNAME, SELECT_ROW_BASE_CLASSNAME } from "../../constants";
import type { SelectionItemProps } from "./types";

function SelectionItemInner(
  { as, active = false, selected = false, disabled = false, children, className, ...props }: SelectionItemProps,
  ref: React.ForwardedRef<HTMLElement>,
) {
  const Component = as as ElementType;
  const resolvedProps = props as Record<string, unknown>;

  return (
    <Component
      ref={ref}
      className={twMerge(
        SELECT_ROW_BASE_CLASSNAME,
        className,
        active || selected ? SELECT_ROW_ACTIVE_CLASSNAME : undefined,
        disabled ? "opacity-50 cursor-not-allowed" : undefined,
      )}
      {...resolvedProps}
    >
      {children}
    </Component>
  );
}

export const SelectionItem = forwardRef(SelectionItemInner) as (
  props: SelectionItemProps & { ref?: React.ForwardedRef<HTMLElement> },
) => React.ReactElement | null;
