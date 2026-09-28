import type { ElementType, MouseEventHandler, ReactNode } from "react";

export type ChatInputModel = {
  id: string;
  name: string;
  logo: ReactNode;
};

export type ChatInputFile = {
  id: string;
  name: string;
  extension: string;
};

export type ChatInputSendPayload = {
  text: string;
  files: ChatInputFile[];
};

type ChatInputLinkAsProps = {
  to: string;
  className?: string;
  children: ReactNode;
  onClick?: MouseEventHandler<HTMLElement>;
};

export type ChatInputLinkAs = ElementType<ChatInputLinkAsProps>;

export type ChatInputMoreModelsLink = {
  href: string;
  linkAs?: ChatInputLinkAs;
  onClick?: () => void;
};

export type ChatInputProps = {
  value: string;
  onChange: (value: string) => void;
  onSend?: (payload: ChatInputSendPayload) => void | Promise<void>;
  canSend?: (payload: ChatInputSendPayload) => boolean;
  onBlockedSend?: (payload: ChatInputSendPayload) => void;
  onAttach?: (file: File) => void | Promise<void>;
  attachAccept?: string;
  placeholder?: string;
  files?: ChatInputFile[];
  onRemoveFile?: (id: string) => void;
  models?: ChatInputModel[];
  activeModelId?: string;
  onModelChange?: (id: string) => void;
  onModelSettingsClick?: () => void;
  moreModelsLink?: ChatInputMoreModelsLink;
  modelPickerDisabled?: boolean;
  streaming?: boolean;
  onStop?: () => void;
  settingsModified?: boolean;
  settingsDisabled?: boolean;
};
