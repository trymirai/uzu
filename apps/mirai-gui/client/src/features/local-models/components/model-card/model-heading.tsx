import type { ReactNode } from "react";
import { Text } from "@/components/ui/typography";

type ModelHeadingProps = {
  name: string;
  logo: ReactNode;
};

export function ModelHeading({ name, logo }: ModelHeadingProps) {
  return (
    <div className="flex items-center gap-3 min-w-0">
      <span className="shrink-0 flex items-center justify-center w-4 h-4 [&>svg]:w-4 [&>svg]:h-4">{logo}</span>
      <Text as="span" size="sm" color="primary" opticalSize={14} className="truncate leading-[1.5] font-[450]">
        {name}
      </Text>
    </div>
  );
}
