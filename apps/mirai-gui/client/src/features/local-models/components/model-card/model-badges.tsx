import { Brain, FileArchive } from "lucide-react";
import { Tooltip } from "@/components/ui/tooltip";
import { Chip, ChipText } from "./chip";

type ModelBadgesProps = {
  parameters?: string;
  size?: string;
  isThinking?: boolean;
  isCompressed?: boolean;
};

export function ModelBadges({ parameters, size, isThinking = false, isCompressed = false }: ModelBadgesProps) {
  return (
    <>
      {isCompressed && (
        <Tooltip content="Compressed model" side="top">
          <Chip className="w-8">
            <FileArchive size={18} className="text-text-muted group-hover/chip:text-text-primary" />
          </Chip>
        </Tooltip>
      )}
      {isThinking && (
        <Tooltip content="Thinking model" side="top">
          <Chip className="w-8">
            <Brain size={18} className="text-text-muted group-hover/chip:text-text-primary" />
          </Chip>
        </Tooltip>
      )}
      {parameters && (
        <Tooltip content="Parameters" side="top">
          <ChipText>{parameters}</ChipText>
        </Tooltip>
      )}
      {size && (
        <Tooltip content="Size" side="top">
          <ChipText>{size}</ChipText>
        </Tooltip>
      )}
    </>
  );
}
