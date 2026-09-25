import type { ReactNode } from "react";
import { cloneElement, isValidElement } from "react";

export function applyIconSize(icon: ReactNode, iconSize: number): ReactNode {
  if (!icon) return icon;
  if (!isValidElement<{ size?: number }>(icon)) return icon;
  if (typeof icon.type === "string") return icon;
  if (icon.props.size !== undefined) return icon;
  return cloneElement(icon, { size: iconSize });
}
