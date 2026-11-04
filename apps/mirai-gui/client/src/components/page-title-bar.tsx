import { type ReactNode, type Ref, useLayoutEffect, useRef, useState } from "react";
import { twMerge } from "tailwind-merge";

/**
 * The title row at the top of a page. Content passes under it through a fade: the page colour at
 * the top melting to clear, with a blur that fades out the same way, so nothing is cut off at a hard
 * edge. The fade stops short of the right edge, where macOS draws its overlay scrollbar, so the
 * scrollbar stays visible over it. Give it to a PageScroller as its header.
 */
export function PageTitleBar({ children, className }: { children: ReactNode; className?: string }) {
  return (
    // Only the row itself takes the pointer: the strip around it belongs to the scroller below, so its
    // scrollbar can still be grabbed at the top.
    <div data-tauri-drag-region="deep" className="pointer-events-none relative isolate">
      <div
        aria-hidden
        className="absolute left-0 right-3 top-0 -bottom-6 -z-10 bg-linear-to-b from-background from-45% to-transparent backdrop-blur-md [mask-image:linear-gradient(to_bottom,black_55%,transparent)]"
      />
      <div
        className={twMerge(
          "pointer-events-auto flex min-h-[72px] items-center gap-3 px-5 pt-7 pb-3 w-full max-w-[calc(var(--chat-width)+2.5rem)] mx-auto min-w-0",
          className,
        )}
      >
        {children}
      </div>
    </div>
  );
}

/**
 * A page that scrolls under a fixed header. The header sits over the scroller rather than inside it,
 * so the rubber-band bounce at either end moves only the content and the header never leaves its
 * place. The scroller is padded by the header's height, for the first content and for focus that
 * scrolls into view.
 */
export function PageScroller({
  header,
  children,
  className,
  scrollRef,
}: {
  header: ReactNode;
  children: ReactNode;
  className?: string;
  scrollRef?: Ref<HTMLDivElement>;
}) {
  const headerRef = useRef<HTMLDivElement>(null);
  const [headerHeight, setHeaderHeight] = useState(72);

  useLayoutEffect(() => {
    const element = headerRef.current;
    if (!element) return;
    const update = () => setHeaderHeight(element.offsetHeight);
    update();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return (
    <div className="relative h-full min-h-0 w-full flex-1">
      <div
        ref={scrollRef}
        className={twMerge("h-full w-full overflow-y-auto overscroll-y-contain page-scrollbar", className)}
        style={{ paddingTop: headerHeight, scrollPaddingTop: headerHeight }}
      >
        {children}
      </div>
      <div ref={headerRef} className="absolute inset-x-0 top-0 z-20">
        {header}
      </div>
    </div>
  );
}
