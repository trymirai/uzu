import { Tooltip } from "@/components/ui/tooltip";
import { ChipText } from "./chip";

type ModelBadgesProps = {
  parameters?: string;
  size?: string;
};

export function ModelBadges({ parameters, size }: ModelBadgesProps) {
  return (
    <>
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
