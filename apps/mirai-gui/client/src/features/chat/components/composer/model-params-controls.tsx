import { getPlatform } from "@/platform/platform-singleton";
import type { SamplingDefaults } from "@/platform/services/chat";
import { STOCHASTIC_SEED, defaultReasoningEffort, useModelParamsStore } from "@/stores/use-model-params-store";
import { useModelsStore } from "@/stores/use-models-store";
import { useRuntimeSessionStore } from "@/stores/use-runtime-session-store";
import type { ReasoningEffort, ReasoningSupport, SamplingPolicyPayload } from "@/types/sampling";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useState } from "react";
import { Toggle } from "@/components/ui/toggle";

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
};

const MODE_OPTIONS = [
  { value: "Default", label: "Default" },
  { value: "Argmax", label: "Argmax" },
  { value: "Stochastic", label: "Stochastic" },
] as const;

const EFFORT_LABELS: Record<ReasoningEffort, string> = {
  disabled: "Off",
  default: "Default",
  low: "Low",
  medium: "Medium",
  high: "High",
  xhigh: "XHigh",
};

const sectionLabel = "text-[12px] font-[450] text-label-muted";

export const ModelParamsControls = ({ repoId }: ModelParamsControlsProps) => {
  const model = useModelsStore((s) => s.models.find((m) => m.repoId === repoId));
  const params = useModelParamsStore((s) => s.paramsByRepoId[repoId] ?? null);
  const globalReasoningEnabled = useModelParamsStore((s) => s.globalReasoningEnabled);
  const setParams = useModelParamsStore((s) => s.setParams);

  const sampling: SamplingPolicyPayload = params?.sampling ?? { type: "Default" };
  const reasoningEffort = params?.reasoningEffort ?? defaultReasoningEffort(globalReasoningEnabled);
  const reasoning: ReasoningSupport = model?.reasoning ?? { kind: model?.isThinking ? "toggle" : "unsupported" };
  const stochastic = sampling.type === "Stochastic" ? sampling : null;

  const isResident = useRuntimeSessionStore((s) => s.residentSession?.repoId === repoId);
  const [modelDefaults, setModelDefaults] = useState<SamplingDefaults | null>(null);
  useEffect(() => setModelDefaults(null), [repoId]);
  useEffect(() => {
    if (!isResident) return;
    let alive = true;
    void getPlatform()
      .chat.getSamplingDefaults(repoId)
      .then((defaults) => {
        if (alive) setModelDefaults(defaults);
      });
    return () => {
      alive = false;
    };
  }, [repoId, isResident]);
  const seed = {
    temperature: modelDefaults?.temperature ?? STOCHASTIC_SEED.temperature,
    topK: modelDefaults?.topK ?? STOCHASTIC_SEED.topK,
    topP: modelDefaults?.topP ?? STOCHASTIC_SEED.topP,
    minP: modelDefaults?.minP ?? STOCHASTIC_SEED.minP,
    repetitionPenalty: modelDefaults?.repetitionPenalty ?? REPETITION_PENALTY_SEED,
    suffixRepetitionLength: modelDefaults?.suffixRepetitionLength ?? SUFFIX_REPETITION_LENGTH_SEED,
  };

  const update = (next: SamplingPolicyPayload) => {
    setParams(repoId, { ...params, sampling: next });
  };

  const onModeChange = (value: string) => {
    if (value === "Stochastic") {
      update({ type: "Stochastic", temperature: seed.temperature, topK: seed.topK, topP: seed.topP, minP: seed.minP });
    } else if (value === "Argmax") {
      update({ type: "Argmax" });
    } else {
      update({ type: "Default" });
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
    update({
      type: "Stochastic",
      temperature: base.temperature ?? seed.temperature,
      topK: base.topK ?? seed.topK,
      ...(base.topP !== undefined ? { topP: base.topP } : {}),
      ...(base.minP !== undefined ? { minP: base.minP } : {}),
      ...(base.repetitionPenalty !== undefined ? { repetitionPenalty: base.repetitionPenalty } : {}),
      ...(base.suffixRepetitionLength !== undefined ? { suffixRepetitionLength: base.suffixRepetitionLength } : {}),
      ...patch,
    });
  };

  const onToggleOptional = (field: "topP" | "minP" | "repetitionPenalty", on: boolean) => {
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

  const onReasoningChange = (effort: ReasoningEffort) => {
    const override = effort === defaultReasoningEffort(globalReasoningEnabled) ? undefined : effort;
    setParams(repoId, { sampling, ...(override ? { reasoningEffort: override } : {}) });
  };

  const levelOptions =
    reasoning.kind === "levels"
      ? reasoning.efforts.map((effort) => ({ value: effort, label: EFFORT_LABELS[effort] }))
      : [];
  const levelValue =
    reasoning.kind === "levels"
      ? reasoning.efforts.includes(reasoningEffort)
        ? reasoningEffort
        : (reasoning.efforts.find((effort) => effort === "default") ??
          reasoning.efforts.find((effort) => effort !== "disabled"))
      : undefined;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1.5">
        <span className={sectionLabel}>Sampling</span>
        <SegmentedControl ariaLabel="Sampling" value={sampling.type} onChange={onModeChange} options={MODE_OPTIONS} />
      </div>

      <AnimatePresence initial={false}>
        {stochastic && (
          <motion.div
            key="stochastic"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="flex flex-col gap-3 pt-3">
              <NumberSliderRow
                label="Temperature"
                min={0}
                max={1}
                step={0.01}
                decimals={2}
                value={stochastic.temperature ?? seed.temperature}
                onChange={(v) => onStochasticChange({ temperature: v })}
              />
              <NumberSliderRow
                label="Top K"
                min={TOP_K_MIN}
                max={TOP_K_MAX}
                step={1}
                decimals={0}
                noSlider
                value={stochastic.topK ?? seed.topK}
                onChange={(v) => onStochasticChange({ topK: v })}
              />
              <NumberSliderRow
                label="Top P"
                min={0}
                max={1}
                step={0.01}
                decimals={2}
                value={stochastic.topP ?? seed.topP}
                enabled={stochastic.topP !== undefined}
                onToggle={(on) => onToggleOptional("topP", on)}
                onChange={(v) => onStochasticChange({ topP: v })}
              />
              <NumberSliderRow
                label="Min P"
                min={0}
                max={1}
                step={0.01}
                decimals={2}
                value={stochastic.minP ?? seed.minP}
                enabled={stochastic.minP !== undefined}
                onToggle={(on) => onToggleOptional("minP", on)}
                onChange={(v) => onStochasticChange({ minP: v })}
              />
              <NumberSliderRow
                label="Repetition penalty"
                min={REPETITION_PENALTY_MIN}
                max={REPETITION_PENALTY_MAX}
                step={0.05}
                decimals={2}
                value={stochastic.repetitionPenalty ?? seed.repetitionPenalty}
                enabled={stochastic.repetitionPenalty !== undefined}
                onToggle={(on) => onToggleOptional("repetitionPenalty", on)}
                onChange={(v) => onStochasticChange({ repetitionPenalty: v })}
              />
              {stochastic.repetitionPenalty !== undefined && (
                <NumberSliderRow
                  label="Suffix repetition length"
                  min={SUFFIX_REPETITION_LENGTH_MIN}
                  max={SUFFIX_REPETITION_LENGTH_MAX}
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

      {reasoning.kind === "toggle" && (
        <div className="flex items-center justify-between gap-3 border-t border-cell-border pt-3">
          <span className="text-[13px] text-label-title">Reasoning</span>
          <Toggle
            checked={reasoningEffort !== "disabled"}
            onChange={(checked) => onReasoningChange(checked ? "default" : "disabled")}
          />
        </div>
      )}

      {reasoning.kind === "levels" && levelValue !== undefined && (
        <div className="flex flex-col gap-1.5 border-t border-cell-border pt-3">
          <span className={sectionLabel}>Reasoning</span>
          <SegmentedControl
            ariaLabel="Reasoning"
            value={levelValue}
            onChange={(value) => onReasoningChange(value as ReasoningEffort)}
            options={levelOptions}
          />
        </div>
      )}
    </div>
  );
};

type NumberSliderRowProps = {
  label: string;
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
        <div className="flex items-center gap-2">
          <span className="text-[13px] text-label-title">{label}</span>
          {optional && <Checkbox checked={active} onChange={(on) => onToggle?.(on)} size="sm" />}
        </div>
        <Input
          size="sm"
          type="text"
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
