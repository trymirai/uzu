import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ElementType, MouseEventHandler, ReactNode } from "react";

export type SelectionItemLinkLikeProps = {
  to: string;
  className?: string;
  children: ReactNode;
  onClick?: MouseEventHandler<HTMLElement>;
};

export type SelectionItemLinkLikeComponent = ElementType<SelectionItemLinkLikeProps>;

type BaseSelectionItemProps = {
  active?: boolean;
  selected?: boolean;
  disabled?: boolean;
  children: ReactNode;
  className?: string;
};

export type SelectionItemButtonProps = BaseSelectionItemProps &
  Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children" | "className" | "disabled"> & {
    as: "button";
  };

export type SelectionItemAnchorProps = BaseSelectionItemProps &
  Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "children" | "className"> & {
    as: "a";
    href: string;
  };

export type SelectionItemLinkProps = BaseSelectionItemProps &
  SelectionItemLinkLikeProps & {
    as: SelectionItemLinkLikeComponent;
  };

export type SelectionItemProps = SelectionItemButtonProps | SelectionItemAnchorProps | SelectionItemLinkProps;
