import { Menu, MenuButton, MenuItem, MenuItems, Transition } from "@headlessui/react";
import { Brain, ChevronDown } from "lucide-react";
import { Fragment } from "react";
import { SelectionItem } from "@/components/ui/select/selection-item";
import { SelectionPanel } from "@/components/ui/select/selection-panel";
import { SelectionSection } from "@/components/ui/select/selection-section";
import { useModelParamsStore } from "@/stores/use-model-params-store";
import { useModelsStore } from "@/stores/use-models-store";
import type { ReasoningEffort, ReasoningSupport } from "@/types/sampling";

const EFFORT_LABELS: Record<ReasoningEffort, string> = {
  default: "Thinking",
  disabled: "Off",
  low: "Low",
  medium: "Medium",
  high: "High",
  xhigh: "XHigh",
};

const EFFORT_ORDER = ["xhigh", "high", "medium", "low", "disabled"] as const;

const availableEfforts = (support: ReasoningSupport | undefined): ReasoningEffort[] => {
  switch (support?.kind) {
    case "toggle":
      return ["default", "disabled"];
    case "levels":
      return EFFORT_ORDER.filter((effort) => support.efforts.includes(effort));
    default:
      return [];
  }
};

type ReasoningSelectorProps = {
  repoId: string;
  disabled?: boolean;
};

export const ReasoningSelector = ({ repoId, disabled = false }: ReasoningSelectorProps) => {
  const support = useModelsStore((s) => s.models.find((model) => model.repoId === repoId)?.reasoning);
  const effort = useModelParamsStore((s) => s.paramsByRepoId[repoId]?.reasoningEffort ?? "default");
  const options = availableEfforts(support);
  if (options.length <= 1) return null;

  const defaultEffort = support?.kind === "toggle" || support?.kind === "levels" ? support.defaultEffort : undefined;
  const selected =
    effort !== "default" && options.includes(effort)
      ? effort
      : defaultEffort && options.includes(defaultEffort)
        ? defaultEffort
        : undefined;
  const label = selected ? EFFORT_LABELS[selected] : "Reasoning";
  const isDefault = selected !== undefined && selected === defaultEffort;
  const onChange = (nextEffort: ReasoningEffort) => {
    const { getParams, setParams } = useModelParamsStore.getState();
    const next = { ...getParams(repoId) };
    if (nextEffort === defaultEffort) delete next.reasoningEffort;
    else next.reasoningEffort = nextEffort;
    setParams(repoId, next);
  };

  return (
    <Menu as="div" className="relative shrink-0">
      <MenuButton
        type="button"
        disabled={disabled}
        aria-label={selected ? `Reasoning: ${label}${isDefault ? ", model default" : ""}` : "Reasoning"}
        title={selected ? `Reasoning: ${label}` : "Reasoning"}
        className="flex h-7 items-center gap-1.5 rounded-md bg-surface-tertiary px-2 text-text-muted text-[13px] font-[450] leading-[1.3] outline-hidden hover:bg-control-surface-hover hover:text-text-primary data-[headlessui-state~=open]:bg-control-surface-active data-[headlessui-state~=open]:text-text-primary data-[focus]:bg-control-surface-hover data-[focus]:text-text-primary data-[focus]:shadow-focus disabled:pointer-events-none disabled:opacity-40"
      >
        <Brain size={14} aria-hidden="true" />
        <span>{label}</span>
        <ChevronDown size={16} aria-hidden="true" />
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
        <div className="absolute right-0 bottom-full pb-1 z-50 origin-bottom-right">
          <MenuItems aria-label="Reasoning" className="outline-hidden">
            <SelectionPanel className="min-w-[160px] pt-1.5">
              <SelectionSection>
                {options.map((option) => (
                  <MenuItem key={option} disabled={disabled}>
                    {({ focus }) => (
                      <SelectionItem
                        as="button"
                        type="button"
                        active={focus}
                        selected={option === selected}
                        aria-current={option === selected ? "true" : undefined}
                        aria-label={`${EFFORT_LABELS[option]}${option === defaultEffort ? ", model default" : ""}`}
                        onClick={() => onChange(option)}
                        className="flex w-full items-center border-none bg-transparent text-left text-text-primary text-[13px] font-[450] leading-[1.3] outline-hidden"
                      >
                        {EFFORT_LABELS[option]}
                        {option === defaultEffort && <DefaultMarker />}
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
};

const DefaultMarker = () => (
  <span
    title="Model default"
    aria-hidden="true"
    className="ml-auto pl-4 text-[11px] font-normal text-text-muted opacity-60"
  >
    default
  </span>
);
