import type { ReactNode } from "react";
import { Text } from "@/components/ui/typography";

type ModelHeadingProps = {
  name: string;
  logo: ReactNode;
  hint?: string;
};

export function ModelHeading({ name, logo, hint }: ModelHeadingProps) {
  return (
    <div className="flex items-center gap-3 min-w-0">
      <span className="shrink-0 flex items-center justify-center w-4 h-4 [&>svg]:w-4 [&>svg]:h-4">{logo}</span>
      <span className="flex flex-col min-w-0">
        <Text as="span" size="sm" color="primary" opticalSize={14} className="truncate leading-[1.5] font-[450]">
          {name}
        </Text>
        {hint && (
          <Text as="span" color="muted" className="truncate text-[12px] leading-[1.3]">
            {hint}
          </Text>
        )}
      </span>
    </div>
  );
}
