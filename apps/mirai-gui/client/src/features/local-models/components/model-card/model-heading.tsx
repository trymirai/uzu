import { splitModelName } from "@/components/model-name";
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
      <Text as="span" size="sm" color="primary" opticalSize={14} className="min-w-0 leading-[1.5] font-[450]">
        {/* The quantization has its own column here, so the name stands alone. */}
        <span className="block truncate" title={name}>
          {splitModelName(name).base}
        </span>
      </Text>
    </div>
  );
}
