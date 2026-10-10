import { Link } from "@tanstack/react-router";
import { ChevronDown } from "lucide-react";
import { Menu, MenuButton, MenuItem, MenuItems, Transition } from "@headlessui/react";
import { forwardRef, Fragment } from "react";
import { SelectionItem } from "@/components/ui/select/selection-item";
import { SelectionPanel } from "@/components/ui/select/selection-panel";
import { SelectionSection } from "@/components/ui/select/selection-section";
import { Text } from "@/components/ui/typography";
import type { ChatInputModel, ChatInputMoreModelsLink } from "./types";

export type ModelPickerProps = {
  models?: ChatInputModel[];
  activeModelId?: string;
  onModelChange?: (id: string) => void;
  moreModelsLink?: ChatInputMoreModelsLink;
  disabled?: boolean;
  menuPlacement?: "up" | "down";
};

const TEXT_13_CLASSNAME = "text-[13px] font-[450] leading-[1.3]";

type SelectionRowProps = {
  active: boolean;
  selected?: boolean;
  children: React.ReactNode;
  onClick?: () => void;
} & Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, "onClick" | "children" | "className">;

// Headless UI tracks menu items through the injected ref/role/id props;
// dropping them breaks keyboard navigation.
const SelectionRow = forwardRef<HTMLElement, SelectionRowProps>(function SelectionRow(
  { active, selected, children, onClick, ...rest },
  ref,
) {
  return (
    <SelectionItem
      as="button"
      type="button"
      ref={ref}
      active={active}
      selected={selected}
      className="flex items-center gap-2 w-full border-none outline-hidden bg-transparent text-text-primary"
      {...rest}
      onClick={onClick}
    >
      {children}
    </SelectionItem>
  );
});

export function ModelPicker({
  models,
  activeModelId,
  onModelChange,
  moreModelsLink,
  disabled = false,
  menuPlacement = "up",
}: ModelPickerProps) {
  const activeModel = models?.find((model) => model.id === activeModelId);

  if ((!models || models.length === 0) && !moreModelsLink) {
    return null;
  }

  return (
    <Menu as="div" className="relative min-w-0 max-w-full">
      {({ close }) => (
        <div className="relative">
          <MenuButton
            as="button"
            type="button"
            disabled={disabled}
            className="group/model flex items-center gap-2 h-7 px-2 max-w-full rounded-md bg-surface-tertiary hover:bg-control-surface-hover data-[headlessui-state~=open]:bg-control-surface-active data-[focus]:bg-control-surface-hover cursor-pointer border-none outline-hidden data-[focus]:shadow-focus disabled:pointer-events-none disabled:opacity-40"
          >
            {activeModel && (
              <span className="shrink-0 flex items-center justify-center w-3.5 h-3.5 [&>svg]:w-3.5 [&>svg]:h-3.5">
                {activeModel.logo}
              </span>
            )}
            <Text
              as="span"
              color="muted"
              opticalSize={14}
              className={`min-w-0 truncate ${TEXT_13_CLASSNAME} group-hover/model:text-text-primary group-data-[focus]/model:text-text-primary group-data-[headlessui-state~=open]/model:text-text-primary`}
            >
              {activeModel ? activeModel.name : "Select model"}
            </Text>
            <ChevronDown
              size={16}
              className="shrink-0 text-text-muted group-hover/model:text-text-primary group-data-[focus]/model:text-text-primary group-data-[headlessui-state~=open]/model:text-text-primary"
            />
          </MenuButton>

          <Transition
            as={Fragment}
            enter="transition-[opacity,scale] duration-[120ms] ease-spring"
            enterFrom="opacity-0 scale-90"
            enterTo="opacity-100 scale-100"
            leave="transition-[opacity,scale] duration-[80ms] ease-spring"
            leaveFrom="opacity-100 scale-100"
            leaveTo="opacity-0 scale-90"
          >
            <div
              className={
                menuPlacement === "up"
                  ? "absolute right-0 bottom-full pb-1 z-50 origin-bottom-right"
                  : "absolute right-0 top-full pt-1 z-50 origin-top-right"
              }
            >
              <MenuItems className="outline-hidden">
                <SelectionPanel className="pt-1.5 min-w-[240px]">
                  {/* Reserve the scrollbar width before WebKit measures the popup. */}
                  <SelectionSection className="max-h-[280px] overflow-y-scroll overscroll-y-contain thin-scrollbar">
                    {models?.map((model) => (
                      <MenuItem key={model.id}>
                        {({ focus }) => (
                          <SelectionRow
                            active={focus}
                            selected={model.id === activeModelId}
                            onClick={() => {
                              onModelChange?.(model.id);
                              close();
                            }}
                          >
                            <span className="shrink-0 flex items-center justify-center w-4 h-4 [&>svg]:w-4 [&>svg]:h-4">
                              {model.logo}
                            </span>
                            <Text
                              as="span"
                              color="primary"
                              opticalSize={14}
                              className={`whitespace-nowrap ${TEXT_13_CLASSNAME}`}
                            >
                              {model.name}
                            </Text>
                          </SelectionRow>
                        )}
                      </MenuItem>
                    ))}
                  </SelectionSection>

                  {moreModelsLink && (
                    <SelectionSection divided className="py-1.5">
                      <MenuItem>
                        {({ focus }) => (
                          <SelectionItem
                            as={Link}
                            to={moreModelsLink.href}
                            active={focus}
                            onClick={() => {
                              moreModelsLink.onClick?.();
                              close();
                            }}
                            className="flex items-center w-full bg-transparent text-text-primary"
                          >
                            <Text as="span" color="primary" opticalSize={14} className={TEXT_13_CLASSNAME}>
                              Download more models
                            </Text>
                          </SelectionItem>
                        )}
                      </MenuItem>
                    </SelectionSection>
                  )}
                </SelectionPanel>
              </MenuItems>
            </div>
          </Transition>
        </div>
      )}
    </Menu>
  );
}
