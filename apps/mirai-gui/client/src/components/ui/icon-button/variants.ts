export type IconButtonVariant = "secondary" | "pill";

export const ICON_BUTTON_VARIANT_MAP: Record<IconButtonVariant, string> = {
  secondary: "bg-gray-50 text-gray-1200 shadow-custom hover:bg-gray-200 hover:shadow-custom-hover active:scale-[0.97]",
  pill: "bg-transparent text-gray-1000 hover:text-gray-1200 active:scale-[0.97]",
};
