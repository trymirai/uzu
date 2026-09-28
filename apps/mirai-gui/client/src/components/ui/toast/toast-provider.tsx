import { Toaster } from "react-hot-toast";

export function ToastProvider() {
  return (
    <Toaster
      position="top-right"
      toastOptions={{ duration: 3000 }}
      containerStyle={{
        top: 20,
        right: 20,
        bottom: 20,
        left: 20,
      }}
    />
  );
}
