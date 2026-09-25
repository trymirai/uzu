import { File as FileIcon, Plus } from "lucide-react";
import { Button } from "../../Button";
import { PopoverMenu } from "../../Popover";
import { Text } from "../../Typography";

export type AttachMenuProps = {
  onSelectFile: () => void;
};

export function AttachMenu({ onSelectFile }: AttachMenuProps) {
  return (
    <PopoverMenu
      side="top"
      align="start"
      sideOffsetPx={4}
      className="min-w-[280px]"
      itemClassName="gap-3 px-2 py-2"
      trigger={
        <Button
          size="xs"
          kind="ghost"
          iconOnly
          ariaLabel="Attach file"
          icon={<Plus size={18} />}
          className="bg-surface-tertiary text-text-muted hover:bg-control-surface-hover hover:text-text-primary"
        />
      }
      items={[
        {
          id: "upload",
          icon: <FileIcon size={16} className="text-text-muted shrink-0" />,
          label: (
            <span>
              <Text as="span" color="primary" opticalSize={14} className="block text-[13px] font-[450] leading-[1.3]">
                Upload a file
              </Text>
              <Text as="span" color="muted" className="block text-[11px] leading-[1.4]">
                Supports TXT, MD, JSON, CSV, YAML
              </Text>
            </span>
          ),
          onClick: onSelectFile,
        },
      ]}
    />
  );
}
