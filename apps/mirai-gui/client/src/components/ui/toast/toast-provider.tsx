import { Toaster } from "react-hot-toast";

export function ToastProvider() {
  return (
    <Toaster
      position="top-right"
      toastOptions={{ duration: 3000 }}
      gutter={8}
      // The chat block sits 8px inside the window with a 12px corner. Toasts sit the same 8px inside the
      // block, with a 4px corner (12 − 8), so their corner runs parallel to the block's.
      containerStyle={{
        top: 16,
        right: 16,
        bottom: 16,
        left: 16,
      }}
    />
  );
}
