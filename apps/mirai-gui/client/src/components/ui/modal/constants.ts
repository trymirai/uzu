// Buttons on a floating surface (dialogs, the update notice): full pills like the composer's, with
// the quiet choice as a translucent fill so it reads on the lifted surface in both themes.
export const SURFACE_BUTTON_CLASSNAME = "rounded-full h-9 px-4 text-[14px] font-[500] active:scale-[0.96]";
export const SURFACE_BUTTON_SMALL_CLASSNAME = "rounded-full h-8 px-3.5 text-[13px] font-[500] active:scale-[0.96]";
export const SURFACE_SOFT_BUTTON_CLASSNAME =
  "bg-black/[0.05] text-text-primary shadow-none hover:bg-black/[0.08] hover:shadow-none active:bg-black/[0.1] dark:bg-white/[0.08] dark:hover:bg-white/[0.12] dark:active:bg-white/[0.14]";

// The round close button in a surface's corner, with a hit area a little larger than it looks.
export const SURFACE_CLOSE_CLASSNAME =
  "relative shrink-0 flex items-center justify-center size-7 rounded-full text-text-muted outline-hidden cursor-pointer transition-colors duration-150 ease-out hover:bg-black/[0.06] hover:text-text-primary dark:hover:bg-white/[0.08] data-[focus]:shadow-focus focus-visible:shadow-focus before:absolute before:-inset-1.5";
