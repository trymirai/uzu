import { Listbox, ListboxButton, ListboxOption, ListboxOptions, Transition } from "@headlessui/react";
import { Check, ChevronDown } from "lucide-react";
import { Fragment } from "react";
import { SelectionItem } from "@/components/ui/select/selection-item";
import { SelectionPanel } from "@/components/ui/select/selection-panel";
import { SelectionSection } from "@/components/ui/select/selection-section";
import type { ThemeMode } from "@/stores/use-app-store";

const THEMES: { value: ThemeMode; label: string }[] = [
  { value: "system", label: "System" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
];

type Props = { value: ThemeMode; onChange: (theme: ThemeMode) => void };

export const ThemeSelector = ({ value, onChange }: Props) => {
  const label = THEMES.find((theme) => theme.value === value)?.label ?? "System";
  return (
    <Listbox value={value} onChange={onChange} as="div" className="relative shrink-0">
      <ListboxButton
        type="button"
        aria-label={`Theme: ${label}`}
        className="flex h-8 min-w-[110px] items-center justify-between gap-3 rounded-md bg-surface-tertiary px-2.5 text-[13px] font-[450] leading-[1.3] text-text-muted outline-hidden hover:bg-control-surface-hover hover:text-text-primary data-[headlessui-state~=open]:bg-control-surface-active data-[headlessui-state~=open]:text-text-primary data-[focus]:bg-control-surface-hover data-[focus]:text-text-primary data-[focus]:shadow-focus"
      >
        <span>{label}</span>
        <ChevronDown size={16} aria-hidden="true" />
      </ListboxButton>
      <Transition
        as={Fragment}
        enter="transition-[opacity,transform] duration-[120ms] ease-spring"
        enterFrom="transform opacity-0 scale-90"
        enterTo="transform opacity-100 scale-100"
        leave="transition-[opacity,transform] duration-[80ms] ease-spring"
        leaveFrom="transform opacity-100 scale-100"
        leaveTo="transform opacity-0 scale-90"
      >
        <div className="absolute right-0 top-full z-50 origin-top-right pt-1">
          <ListboxOptions className="outline-hidden">
            <SelectionPanel className="min-w-[160px] origin-top-right py-1.5">
              <SelectionSection>
                {THEMES.map((theme) => (
                  <ListboxOption key={theme.value} value={theme.value} as={Fragment}>
                    {({ focus, selected }) => (
                      <SelectionItem
                        as="button"
                        type="button"
                        active={focus}
                        selected={selected}
                        className="flex w-full items-center justify-between gap-4 border-none bg-transparent text-left text-[13px] font-[450] leading-[1.3] text-text-primary outline-hidden"
                      >
                        {theme.label}
                        {selected && <Check size={14} aria-hidden="true" />}
                      </SelectionItem>
                    )}
                  </ListboxOption>
                ))}
              </SelectionSection>
            </SelectionPanel>
          </ListboxOptions>
        </div>
      </Transition>
    </Listbox>
  );
};
