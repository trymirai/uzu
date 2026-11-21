import { Menu, MenuButton, MenuItem, MenuItems, Transition } from "@headlessui/react";
import { Check, ChevronDown, Globe } from "lucide-react";
import { Fragment } from "react";
import { twMerge } from "tailwind-merge";
import { SelectionItem } from "@/components/ui/select/selection-item";
import { SelectionPanel } from "@/components/ui/select/selection-panel";
import { SelectionSection } from "@/components/ui/select/selection-section";
import { Tooltip } from "@/components/ui/tooltip";
import type { WebSearchMode } from "@/types/web-search";
import type { ChatInputWebSearch } from "./types";

const MODES: Record<WebSearchMode, { name: string; description: string; buttonLabel?: string }> = {
  off: { name: "Off", description: "The model answers without the web" },
  ask: { name: "Ask each time", description: "Show each search and wait for your answer", buttonLabel: "Ask" },
  auto: { name: "Search without asking", description: "The model searches when it needs to", buttonLabel: "Search" },
};
const MODE_ORDER: WebSearchMode[] = ["off", "ask", "auto"];

const UNAVAILABLE_HINT = "This model cannot search the web";
const BUTTON_CLASSNAME =
  "flex h-7 items-center gap-1.5 rounded-full pl-2.5 pr-2 text-[13px] font-[450] leading-[1.3] outline-hidden transition-colors duration-150 ease-out data-[focus]:shadow-focus disabled:cursor-not-allowed disabled:opacity-50";
const OFF_CLASSNAME = "bg-control-fill text-text-secondary hover:bg-control-fill-hover hover:text-text-primary";
const ON_CLASSNAME = "bg-blue-500/10 text-blue-500 hover:bg-blue-500/15";

export function WebSearchSelector({ mode, onChange, available, disabled = false }: ChatInputWebSearch) {
  const currentMode = available ? mode : "off";
  const current = MODES[currentMode];
  const blocked = disabled || !available;

  return (
    <Menu as="div" className="relative shrink-0">
      <Tooltip content={available ? `Web search: ${current.name}` : UNAVAILABLE_HINT} side="top">
        <MenuButton
          type="button"
          disabled={blocked}
          aria-label={`Web search: ${current.name}`}
          className={twMerge(BUTTON_CLASSNAME, current.buttonLabel ? ON_CLASSNAME : OFF_CLASSNAME)}
        >
          <Globe size={14} aria-hidden="true" />
          {current.buttonLabel && <span>{current.buttonLabel}</span>}
          <ChevronDown size={16} aria-hidden="true" />
        </MenuButton>
      </Tooltip>
      <Transition
        as={Fragment}
        enter="transition-[opacity,scale] duration-200 ease-spring motion-reduce:transition-none"
        enterFrom="opacity-0 scale-95"
        enterTo="opacity-100 scale-100"
        leave="transition-[opacity,scale] duration-[140ms] ease-spring motion-reduce:transition-none"
        leaveFrom="opacity-100 scale-100"
        leaveTo="opacity-0 scale-95"
      >
        <div className="absolute left-0 bottom-full pb-2 z-50 origin-bottom-left">
          <MenuItems aria-label="Web search" className="outline-hidden">
            <SelectionPanel className="w-max min-w-[260px] py-1 origin-bottom-left">
              <SelectionSection>
                {MODE_ORDER.map((option) => (
                  <MenuItem key={option}>
                    {({ focus }) => (
                      <SelectionItem
                        as="button"
                        type="button"
                        active={focus}
                        selected={option === currentMode}
                        aria-current={option === currentMode ? "true" : undefined}
                        onClick={() => onChange(option)}
                        className="flex w-full items-start gap-2.5 border-none bg-transparent text-left outline-hidden"
                      >
                        <span className="min-w-0 flex-1">
                          <span className="block text-[13px] font-[450] leading-[1.3] text-text-primary">
                            {MODES[option].name}
                          </span>
                          <span className="mt-0.5 block text-[12px] leading-[1.35] text-text-muted">
                            {MODES[option].description}
                          </span>
                        </span>
                        <Check
                          size={16}
                          aria-hidden="true"
                          className={twMerge("mt-px shrink-0 text-text-primary", option !== currentMode && "invisible")}
                        />
                      </SelectionItem>
                    )}
                  </MenuItem>
                ))}
              </SelectionSection>
            </SelectionPanel>
          </MenuItems>
        </div>
      </Transition>
    </Menu>
  );
}
