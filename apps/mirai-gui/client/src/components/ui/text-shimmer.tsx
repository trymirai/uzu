import React from "react";
import { motion } from "motion/react";
import { twMerge } from "tailwind-merge";

export type TextShimmerProps = {
  children: string;
  className?: string;
  duration?: number;
  spread?: number;
};

function TextShimmerComponent({ children, className, duration = 2, spread = 2 }: TextShimmerProps) {
  const dynamicSpread = children.length * spread;

  return (
    <motion.p
      className={twMerge(
        "relative inline-block bg-[length:250%_100%,auto] bg-clip-text",
        // The resting colour is the muted label colour in both themes, so shimmering text matches the
        // muted text around it (the reasoning under "Thinking…"); only the moving highlight differs.
        "text-transparent [--base-color:var(--color-label-muted)] [--base-gradient-color:#000]",
        "[background-repeat:no-repeat,padding-box] [--bg:linear-gradient(90deg,#0000_calc(50%-var(--spread)),var(--base-gradient-color),#0000_calc(50%+var(--spread)))]",
        "dark:[--base-gradient-color:#ffffff] dark:[--bg:linear-gradient(90deg,#0000_calc(50%-var(--spread)),var(--base-gradient-color),#0000_calc(50%+var(--spread)))]",
        className,
      )}
      initial={{ backgroundPosition: "100% center" }}
      animate={{ backgroundPosition: "0% center" }}
      transition={{
        repeat: Infinity,
        duration,
        ease: "linear",
      }}
      style={
        {
          "--spread": `${dynamicSpread}px`,
          backgroundImage: `var(--bg), linear-gradient(var(--base-color), var(--base-color))`,
        } as React.CSSProperties
      }
    >
      {children}
    </motion.p>
  );
}

export const TextShimmer = React.memo(TextShimmerComponent);
