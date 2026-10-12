import type { SamplingDefaults } from "@/platform/services/chat";
import { resolveModelTools, useModelParamsStore } from "@/stores/use-model-params-store";
import { useModelsStore } from "@/stores/use-models-store";
import { TEMPERATURE_MIN, type SamplingPolicyPayload } from "@/types/sampling";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useState, useId } from "react";
import { Toggle } from "@/components/ui/toggle";
import { RotateCcw } from "lucide-react";
import { Button } from "@headlessui/react";

const TOP_K_MIN = 1;
const TOP_K_MAX = 4096;
const REPETITION_PENALTY_MIN = 1;
const REPETITION_PENALTY_MAX = 2;
const REPETITION_PENALTY_SEED = 1.1;
const SUFFIX_REPETITION_LENGTH_MIN = 32;
const SUFFIX_REPETITION_LENGTH_MAX = 4096;
const SUFFIX_REPETITION_LENGTH_STEP = 32;
const SUFFIX_REPETITION_LENGTH_SEED = 32;

type ModelParamsControlsProps = {
  repoId: string;
  samplingDefaults: SamplingDefaults | null;
};

const MODE_OPTIONS = [
  { value: "Stochastic", label: "Stochastic" },
  { value: "Greedy", label: "Greedy" },
] as const;

type StochasticSampling = Extract<SamplingDefaults, { type: "Stochastic" }>;
type SamplingField = Exclude<keyof StochasticSampling, "type">;

const samplingValue = (sampling: StochasticSampling, field: SamplingField) =>
  sampling[field] ?? (field === "temperature" ? 1 : null);

const sameSampling = (left: SamplingDefaults, right: SamplingDefaults) =>
  left.type === right.type &&
  (left.type === "Greedy" ||
    (right.type === "Stochastic" &&
      (["temperature", "topK", "topP", "minP", "repetitionPenalty", "suffixRepetitionLength"] as const).every(
        (field) => samplingValue(left, field) === samplingValue(right, field),
      )));

const ChangedMarker = () => (
  <span aria-hidden="true" title="Changed from default" className="ml-0.5 text-red-500">
    *
  </span>
);

const SectionHeading = ({ name, changed, onReset }: { name: string; changed: boolean; onReset: () => void }) => (
  <div className="flex h-5 items-center gap-1">
    <span className="text-[12px] font-[450] text-label-muted">{name}</span>
    {changed && (
      <Button
        type="button"
        onClick={onReset}
        aria-label={`Reset ${name.toLowerCase()} to defaults`}
        title={`Reset ${name.toLowerCase()} to defaults`}
        className="inline-flex size-5 items-center justify-center rounded text-red-500 transition-colors hover:bg-red-500/10 outline-hidden data-[focus]:shadow-focus"
      >
        <RotateCcw aria-hidden="true" className="size-3" />
      </Button>
    )}
  </div>
);

export const ModelParamsControls = ({ repoId, samplingDefaults: modelDefaults }: ModelParamsControlsProps) => {
  const model = useModelsStore((s) => s.models.find((m) => m.repoId === repoId));
  const params = useModelParamsStore((s) => s.paramsByRepoId[repoId] ?? null);
  const globalModelChatNamingEnabled = useModelParamsStore((s) => s.globalModelChatNamingEnabled);
  const setParams = useModelParamsStore((s) => s.setParams);

  const sampling: SamplingPolicyPayload = params?.sampling ?? { type: "Default" };
  const tools = resolveModelTools(params ?? undefined, globalModelChatNamingEnabled, model?.paramSize);
  const defaultTools = resolveModelTools(undefined, globalModelChatNamingEnabled, model?.paramSize);
  const supportsTools = model?.supportsTools === true;

  const resolvedSampling = sampling.type === "Default" ? modelDefaults : sampling;
  const stochastic = resolvedSampling?.type === "Stochastic" ? resolvedSampling : null;
  const defaultStochastic = modelDefaults?.type === "Stochastic" ? modelDefaults : null;
  const samplingChanged = sampling.type !== "Default" && (!modelDefaults || !sameSampling(sampling, modelDefaults));
  const modeChanged = modelDefaults !== null && resolvedSampling?.type !== modelDefaults.type;
  const fieldChanged = (field: SamplingField) =>
    stochastic !== null &&
    defaultStochastic !== null &&
    samplingValue(stochastic, field) !== samplingValue(defaultStochastic, field);
  const dateTimeChanged = tools.dateTimeToolEnabled !== defaultTools.dateTimeToolEnabled;
  const chartChanged = tools.chartToolEnabled !== defaultTools.chartToolEnabled;
  const chatNamingChanged = tools.modelChatNamingEnabled !== defaultTools.modelChatNamingEnabled;
  // Seeds are only for explicitly enabling a filter; missing parameters stay disabled.
  const seed = {
    topK: defaultStochastic?.topK ?? 40,
    topP: defaultStochastic?.topP ?? 0.95,
    minP: defaultStochastic?.minP ?? 0.05,
    repetitionPenalty: defaultStochastic?.repetitionPenalty ?? REPETITION_PENALTY_SEED,
    suffixRepetitionLength: defaultStochastic?.suffixRepetitionLength ?? SUFFIX_REPETITION_LENGTH_SEED,
  };

  const update = (next: SamplingDefaults) => {
    setParams(repoId, {
      ...params,
      sampling: modelDefaults && sameSampling(next, modelDefaults) ? { type: "Default" } : next,
    });
  };

  const onModeChange = (value: string) => {
    if (value === "Stochastic") {
      update(defaultStochastic ?? { type: "Stochastic" });
    } else if (value === "Greedy") {
      update({ type: "Greedy" });
    }
  };

  const onStochasticChange = (
    patch: Partial<{
      temperature: number;
      topK: number;
      topP: number;
      minP: number;
      repetitionPenalty: number;
      suffixRepetitionLength: number;
    }>,
  ) => {
    const base = stochastic ?? { type: "Stochastic" as const };
    update({ ...base, ...patch });
  };

  const onToggleOptional = (field: "topK" | "topP" | "minP" | "repetitionPenalty", on: boolean) => {
    if (on) {
      const patch =
        field === "repetitionPenalty"
          ? { repetitionPenalty: seed.repetitionPenalty, suffixRepetitionLength: seed.suffixRepetitionLength }
          : { [field]: seed[field] };
      onStochasticChange(patch);
    } else {
      const next = { ...(stochastic ?? { type: "Stochastic" as const }) };
      delete next[field];
      if (field === "repetitionPenalty") delete next.suffixRepetitionLength;
      update({ ...next, type: "Stochastic" });
    }
  };

  const onToolChange = (
    field: "modelChatNamingEnabled" | "dateTimeToolEnabled" | "chartToolEnabled",
    enabled: boolean,
  ) => {
    const next = { ...params, sampling };
    if (enabled === defaultTools[field]) delete next[field];
    else next[field] = enabled;
    setParams(repoId, next);
  };

  const resetSampling = () => setParams(repoId, { ...params, sampling: { type: "Default" } });
  const resetTools = () => {
    const next = { ...params, sampling };
    delete next.dateTimeToolEnabled;
    delete next.chartToolEnabled;
    if (globalModelChatNamingEnabled) delete next.modelChatNamingEnabled;
    setParams(repoId, next);
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1.5">
        <SectionHeading name="Sampling" changed={samplingChanged} onReset={resetSampling} />
        <SegmentedControl
          ariaLabel="Sampling"
          value={resolvedSampling?.type ?? ""}
          onChange={onModeChange}
          options={MODE_OPTIONS.map((option) => ({
            ...option,
            ariaLabel: option.label,
            label: (
              <span>
                {option.label}
                {modeChanged && option.value === resolvedSampling?.type && <ChangedMarker />}
              </span>
            ),
          }))}
        />
        {!resolvedSampling && <p className="text-[12px] text-label-muted">Model sampling settings are unavailable.</p>}
      </div>

      <AnimatePresence initial={false}>
        {stochastic && (
          <motion.div
            key="stochastic"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeInOut" }}
            className="overflow-clip"
          >
            <div className="flex flex-col gap-3 pt-3">
              <NumberSliderRow
                label="Temperature"
                changed={fieldChanged("temperature")}
                min={Math.min(TEMPERATURE_MIN, stochastic.temperature ?? 1)}
                max={Math.max(2, stochastic.temperature ?? 1)}
                step={0.01}
                decimals={2}
                value={stochastic.temperature ?? 1}
                onChange={(v) => onStochasticChange({ temperature: v })}
              />
              <NumberSliderRow
                label="Top K"
                changed={fieldChanged("topK")}
                min={TOP_K_MIN}
                max={Math.max(TOP_K_MAX, stochastic.topK ?? 0)}
                step={1}
                decimals={0}
                noSlider
                value={stochastic.topK ?? seed.topK}
                enabled={stochastic.topK != null}
                onToggle={(on) => onToggleOptional("topK", on)}
                onChange={(v) => onStochasticChange({ topK: v })}
              />
              <NumberSliderRow
                label="Top P"
                changed={fieldChanged("topP")}
                min={0}
                max={1}
                step={0.01}
                decimals={2}
                value={stochastic.topP ?? seed.topP}
                enabled={stochastic.topP != null}
                onToggle={(on) => onToggleOptional("topP", on)}
                onChange={(v) => onStochasticChange({ topP: v })}
              />
              <NumberSliderRow
                label="Min P"
                changed={fieldChanged("minP")}
                min={0}
                max={1}
                step={0.01}
                decimals={2}
                value={stochastic.minP ?? seed.minP}
                enabled={stochastic.minP != null}
                onToggle={(on) => onToggleOptional("minP", on)}
                onChange={(v) => onStochasticChange({ minP: v })}
              />
              <NumberSliderRow
                label="Repetition penalty"
                changed={fieldChanged("repetitionPenalty")}
                min={Math.min(REPETITION_PENALTY_MIN, stochastic.repetitionPenalty ?? REPETITION_PENALTY_MIN)}
                max={Math.max(REPETITION_PENALTY_MAX, stochastic.repetitionPenalty ?? REPETITION_PENALTY_MAX)}
                step={0.05}
                decimals={2}
                value={stochastic.repetitionPenalty ?? seed.repetitionPenalty}
                enabled={stochastic.repetitionPenalty != null}
                onToggle={(on) => onToggleOptional("repetitionPenalty", on)}
                onChange={(v) => onStochasticChange({ repetitionPenalty: v })}
              />
              {stochastic.repetitionPenalty != null && (
                <NumberSliderRow
                  label="Suffix repetition length"
                  changed={fieldChanged("suffixRepetitionLength")}
                  min={Math.min(
                    SUFFIX_REPETITION_LENGTH_MIN,
                    stochastic.suffixRepetitionLength ?? SUFFIX_REPETITION_LENGTH_MIN,
                  )}
                  max={Math.max(
                    SUFFIX_REPETITION_LENGTH_MAX,
                    stochastic.suffixRepetitionLength ?? SUFFIX_REPETITION_LENGTH_MAX,
                  )}
                  step={SUFFIX_REPETITION_LENGTH_STEP}
                  decimals={0}
                  noSlider
                  value={stochastic.suffixRepetitionLength ?? seed.suffixRepetitionLength}
                  onChange={(v) => onStochasticChange({ suffixRepetitionLength: v })}
                />
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex flex-col gap-3 border-t border-cell-border pt-3">
        <SectionHeading
          name="Tools"
          changed={dateTimeChanged || chartChanged || chatNamingChanged}
          onReset={resetTools}
        />
        {!supportsTools && (
          <p className="text-[12px] leading-relaxed text-label-muted">This model does not support tool calls.</p>
        )}
        <div className="flex items-start justify-between gap-3">
          <div className="flex flex-col gap-1">
            <span className="text-[13px] text-label-title">
              Current date and time{dateTimeChanged && <ChangedMarker />}
            </span>
            <p className="text-[12px] leading-relaxed text-label-muted">Look up the current local date and time.</p>
          </div>
          <Toggle
            label="Current date and time"
            checked={supportsTools && tools.dateTimeToolEnabled}
            disabled={!supportsTools}
            onChange={(enabled) => onToolChange("dateTimeToolEnabled", enabled)}
          />
        </div>
        <div className="flex items-start justify-between gap-3">
          <div className="flex flex-col gap-1">
            <span className="text-[13px] text-label-title">Draw charts{chartChanged && <ChangedMarker />}</span>
            <p className="text-[12px] leading-relaxed text-label-muted">Show charts in replies.</p>
          </div>
          <Toggle
            label="Draw charts"
            checked={supportsTools && tools.chartToolEnabled}
            disabled={!supportsTools}
            onChange={(enabled) => onToolChange("chartToolEnabled", enabled)}
          />
        </div>
        {globalModelChatNamingEnabled && (
          <div className="flex items-start justify-between gap-3">
            <div className="flex flex-col gap-1">
              <span className="text-[13px] text-label-title">Name chat{chatNamingChanged && <ChangedMarker />}</span>
              <p className="text-[12px] leading-relaxed text-label-muted">
                Name and rename the chat using a tool call.
              </p>
            </div>
            <Toggle
              label="Name chat using a tool"
              checked={supportsTools && tools.modelChatNamingEnabled}
              disabled={!supportsTools}
              onChange={(enabled) => onToolChange("modelChatNamingEnabled", enabled)}
            />
          </div>
        )}
      </div>
    </div>
  );
};

type NumberSliderRowProps = {
  label: string;
  changed: boolean;
  min: number;
  max: number;
  step: number;
  decimals: number;
  value: number;
  onChange: (value: number) => void;
  enabled?: boolean;
  onToggle?: (on: boolean) => void;
  noSlider?: boolean;
};

const fmt = (value: number, decimals: number) => (decimals > 0 ? String(value) : String(Math.round(value)));

const NumberSliderRow = ({
  label,
  changed,
  min,
  max,
  step,
  decimals,
  value,
  onChange,
  enabled,
  onToggle,
  noSlider,
}: NumberSliderRowProps) => {
  const optional = onToggle !== undefined;
  const active = !optional || enabled === true;

  const labelId = useId();
  const [draft, setDraft] = useState(fmt(value, decimals));
  useEffect(() => {
    // Keep what the user typed while it still parses to the value ("1." for 1).
    setDraft((current) => (Number(current) === value ? current : fmt(value, decimals)));
  }, [value, decimals]);

  const onInput = (raw: string) => {
    setDraft(raw);
    if (raw.trim() === "") return;
    const n = decimals > 0 ? Number(raw) : Math.floor(Number(raw));
    if (Number.isFinite(n) && n >= min && n <= max) onChange(n);
  };

  const onBlur = () => {
    const parsed = decimals > 0 ? Number(draft) : Math.floor(Number(draft));
    const next = Number.isFinite(parsed) ? Math.min(max, Math.max(min, parsed)) : value;
    setDraft(fmt(next, decimals));
    if (next !== value) onChange(next);
  };

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center">
          <div className="flex items-center gap-2">
            <span id={labelId} className="text-[13px] text-label-title">
              {label}
            </span>
            {optional && (
              <Checkbox checked={active} onChange={(on) => onToggle?.(on)} size="sm" aria-label={`Enable ${label}`} />
            )}
          </div>
          {changed && <ChangedMarker />}
        </div>
        <Input
          size="sm"
          type="text"
          aria-labelledby={labelId}
          inputMode="decimal"
          disabled={!active}
          className="w-20"
          value={active ? draft : ""}
          onChange={(e) => onInput(e.target.value)}
          onBlur={onBlur}
        />
      </div>
      {active && !noSlider && (
        <input
          type="range"
          aria-labelledby={labelId}
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="my-1 h-1.5 w-full cursor-pointer appearance-none rounded-full bg-slider-track [&::-webkit-slider-thumb]:size-3 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-text-primary"
        />
      )}
    </div>
  );
};
