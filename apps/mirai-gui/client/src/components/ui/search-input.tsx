import { Search } from "lucide-react";
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
      size="sm"
      fullWidth
      leftIcon={<Search size={14} />}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className={className}
    />
  );
}
