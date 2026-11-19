import { Listbox, ListboxButton, ListboxOption, ListboxOptions, Transition } from "@headlessui/react";
import { Check, ChevronDown } from "lucide-react";
import { Fragment } from "react";
import { SelectionItem } from "./selection-item";
import { SelectionPanel } from "./selection-panel";
import { SelectionSection } from "./selection-section";

type PillSelectProps<Value extends string> = {
  label: string;
  value: Value;
  options: readonly { value: Value; label: string }[];
  onChange: (value: Value) => void;
};

export const PillSelect = <Value extends string>({ label, value, options, onChange }: PillSelectProps<Value>) => {
  const selected = options.find((option) => option.value === value)?.label ?? value;
  return (
    <Listbox value={value} onChange={onChange} as="div" className="relative shrink-0">
      <ListboxButton
        type="button"
        aria-label={`${label}: ${selected}`}
        className="flex h-8 min-w-[110px] items-center justify-between gap-3 rounded-full bg-control-fill px-3 text-[13px] font-[450] leading-[1.3] text-text-primary outline-hidden transition-colors duration-150 ease-out hover:bg-control-fill-hover data-[headlessui-state~=open]:bg-control-fill-hover data-[focus]:bg-control-fill-hover data-[focus]:shadow-focus"
      >
        <span>{selected}</span>
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
            <SelectionPanel className="min-w-[160px] origin-top-right py-1">
              <SelectionSection>
                {options.map((option) => (
                  <ListboxOption key={option.value} value={option.value} as={Fragment}>
                    {({ focus, selected }) => (
                      <SelectionItem
                        as="button"
                        type="button"
                        active={focus}
                        selected={selected}
                        className="flex w-full items-center justify-between gap-4 border-none bg-transparent text-left text-[13px] font-[450] leading-[1.3] text-text-primary outline-hidden"
                      >
                        {option.label}
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
