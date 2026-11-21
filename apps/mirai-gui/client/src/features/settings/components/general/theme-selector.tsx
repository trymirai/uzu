import type { ThemeMode } from "@/stores/use-app-store";
import { PillSelect } from "@/components/ui/select/pill-select";

const THEMES: { value: ThemeMode; label: string }[] = [
  { value: "system", label: "System" },
  { value: "dark", label: "Dark" },
  { value: "light", label: "Light" },
];

type Props = { value: ThemeMode; onChange: (theme: ThemeMode) => void };

export const ThemeSelector = ({ value, onChange }: Props) => (
  <PillSelect label="Theme" value={value} options={THEMES} onChange={onChange} />
);
