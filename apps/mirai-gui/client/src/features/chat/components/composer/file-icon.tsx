import type { ComponentType } from "react";
import { File, Table } from "lucide-react";

import { CodeIcon } from "@/components/icons/code-icon";

type FileIconProps = {
  extension: string;
  className?: string;
};

const ICON_BY_EXTENSION: Record<string, ComponentType<{ className?: string }>> = {
  txt: File,
  md: File,
  json: CodeIcon,
  csv: Table,
  tsv: Table,
  py: CodeIcon,
  js: CodeIcon,
  ts: CodeIcon,
  tsx: CodeIcon,
  jsx: CodeIcon,
  html: CodeIcon,
  css: CodeIcon,
  xml: CodeIcon,
  yaml: CodeIcon,
  yml: CodeIcon,
};

export const FileIcon = ({ extension, className }: FileIconProps) => {
  const IconComponent = ICON_BY_EXTENSION[extension] ?? File;
  return <IconComponent className={className} />;
};
