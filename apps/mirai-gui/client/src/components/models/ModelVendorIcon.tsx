import { useAppStore } from "@/stores/useAppStore";
import { useModelsStore } from "@/stores/useModelsStore";
import { VendorIcon } from "@/ui-kit";

type ModelVendorIconProps = {
  vendor?: string | null;
  className?: string;
  size?: number;
};

export const ModelVendorIcon = ({ vendor, className = "h-4 w-4", size = 16 }: ModelVendorIconProps) => {
  const isDarkMode = useAppStore((s) => s.isDarkMode);
  const icons = useModelsStore((s) => (vendor ? s.vendorIconsByName[vendor] : undefined));
  if (!vendor || !icons) return null;
  return <VendorIcon src={isDarkMode ? icons.dark : icons.light} vendor={vendor} className={className} size={size} />;
};
