import { Search } from "lucide-react";
import { twMerge } from "tailwind-merge";
import { Input } from "./input";

export type SearchInputProps = {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
};

export function SearchInput({ value, onChange, placeholder = "Search", className }: SearchInputProps) {
  return (
    <Input
      size="md"
      fullWidth
      leftIcon={<Search size={15} />}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      // A filled pill rather than an outlined box: it sits on the page like the other controls.
      className={twMerge(
        "h-8 rounded-full border-transparent bg-control-fill hover:border-transparent focus-within:border-border-outlined",
        className,
      )}
    />
  );
}
