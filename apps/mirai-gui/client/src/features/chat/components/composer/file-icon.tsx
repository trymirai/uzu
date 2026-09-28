import React from "react";
import { File, Table } from "lucide-react";

import { getFileIcon } from "../../services/attachment-files";
import { CodeIcon } from "@/components/icons/code-icon";

type FileIconProps = {
  extension: string;
  className?: string;
};

export const FileIcon: React.FC<FileIconProps> = ({ extension, className }) => {
  const iconName = getFileIcon(extension);

  const iconMap = {
    File: File,
    FileText: File,
    Code: CodeIcon,
    Table: Table,
  };

  const IconComponent = iconMap[iconName as keyof typeof iconMap] || File;

  return <IconComponent className={className} />;
};
