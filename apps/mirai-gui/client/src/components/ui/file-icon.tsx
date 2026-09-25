import React from "react";
import { File, Table } from "lucide-react";

import { getFileIcon } from "../../utils/file-utils";
import { CodeIcon } from "@/components/icons/CodeIcon";

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
