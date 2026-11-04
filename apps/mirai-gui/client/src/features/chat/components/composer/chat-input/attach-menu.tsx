import { IconPageText, IconPlusMedium } from "@central-icons-react/round-outlined-radius-3-stroke-2";
import { Button } from "@/components/ui/button";
import { PopoverMenu } from "@/components/ui/popover/popover-menu";
import { Text } from "@/components/ui/typography";

export type AttachMenuProps = {
  onSelectFile: () => void;
};

export function AttachMenu({ onSelectFile }: AttachMenuProps) {
  return (
    <PopoverMenu
      side="top"
      align="start"
      sideOffsetPx={8}
      className="min-w-[260px]"
      itemClassName="gap-3 px-2.5 py-2"
      trigger={
        <Button
          size="xs"
          kind="ghost"
          iconOnly
          ariaLabel="Attach file"
          icon={<IconPlusMedium size={18} aria-hidden />}
          className="rounded-full bg-control-fill text-text-secondary hover:bg-control-fill-hover hover:text-text-primary"
        />
      }
      items={[
        {
          id: "upload",
          icon: <IconPageText size={18} className="text-text-muted shrink-0" aria-hidden />,
          label: (
            <span>
              <Text as="span" color="primary" className="block text-[13px] font-[500] leading-[1.3]">
                Upload a file
              </Text>
              <Text as="span" color="muted" className="mt-0.5 block text-[12px] leading-[1.35]">
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
