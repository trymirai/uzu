export type ToastType = "success" | "error" | "info" | "warning";

export type ToastOptions = {
  id?: string;
  onClick?: () => void;
  /** A quieter second line: what to do next, or the details of an error. */
  description?: string;
};
