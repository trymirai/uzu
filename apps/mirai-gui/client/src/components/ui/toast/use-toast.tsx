import toast from "react-hot-toast";
import { CustomToast } from "./custom-toast";
import type { ToastOptions, ToastType } from "./types";

function showToast(message: string, type: ToastType, options?: ToastOptions) {
  return toast.custom((t) => <CustomToast t={t} message={message} type={type} onClick={options?.onClick} />, {
    id: options?.id,
    duration: 4000,
    position: "top-right",
  });
}

const toastApi = {
  success: (message: string, options?: ToastOptions) => showToast(message, "success", options),
  error: (message: string, options?: ToastOptions) => showToast(message, "error", options),
  info: (message: string, options?: ToastOptions) => showToast(message, "info", options),
  warning: (message: string, options?: ToastOptions) => showToast(message, "warning", options),
  dismiss: toast.dismiss,
};

export type ToastApi = typeof toastApi;

export function useToast(): ToastApi {
  return toastApi;
}
