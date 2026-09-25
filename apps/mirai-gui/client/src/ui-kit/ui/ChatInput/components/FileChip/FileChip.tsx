import { File as FileIcon, X } from "lucide-react";
import { useState } from "react";
import { twMerge } from "tailwind-merge";
import { Button } from "../../../Button";
import { Text } from "../../../Typography";
import type { ChatInputFile } from "../../types";

const EXT_COLORS: Record<string, string> = {
  txt: "text-text-muted",
  md: "text-text-muted",
  json: "text-amber-500",
  csv: "text-green-500",
  yaml: "text-blue-500",
  yml: "text-blue-500",
};

export type FileChipProps = {
  file: ChatInputFile;
  onRemove: () => void;
};

export function FileChip({ file, onRemove }: FileChipProps) {
  const [hovered, setHovered] = useState(false);
  const extColor = EXT_COLORS[file.extension.toLowerCase()] ?? "text-text-muted";

  return (
    <div
      className="relative group flex items-center gap-2 rounded-md bg-surface-tertiary px-2 py-2 min-w-[80px]"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {hovered && (
        <div className="absolute -top-1.5 -right-1.5">
          <Button
            size="xxs"
            kind="ghost"
            iconOnly
            ariaLabel="Remove file"
            icon={<X size={10} />}
            onClick={(e) => {
              e.stopPropagation();
              onRemove();
            }}
            className="size-5 rounded-full bg-surface-elevated shadow-sm text-text-muted hover:text-text-primary hover:bg-surface-elevated"
          />
        </div>
      )}
      <FileIcon size={14} className="text-text-muted shrink-0" />
      <Text
        as="span"
        color="primary"
        opticalSize={14}
        className="truncate max-w-[160px] text-[13px] font-[450] leading-[1.3]"
      >
        {file.name}
      </Text>
      <Text as="span" opticalSize={14} className={twMerge("text-[13px] font-[450] leading-[1.3]", extColor)}>
        {file.extension.toLowerCase()}
      </Text>
    </div>
  );
}
