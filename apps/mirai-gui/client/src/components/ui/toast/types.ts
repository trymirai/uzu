export type ToastType = "success" | "error" | "info" | "warning";

export type ToastOptions = {
  id?: string;
  onClick?: () => void;
};
