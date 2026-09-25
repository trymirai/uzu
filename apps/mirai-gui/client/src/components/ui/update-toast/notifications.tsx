import { UpdateToast } from "@/components/ui/update-toast";
import { toast } from "react-hot-toast";

const toastPosition = "bottom-right" as const;

type ReadyToastParams = {
  version: string;
  toastId: string;
  onApplyNow: () => void | Promise<void>;
  onLater: () => void | Promise<void>;
  errorMessage?: string;
};

export const dismissUpdateToast = (toastId: string): void => {
  toast.dismiss(toastId);
};

export const showReadyUpdateToast = ({
  version,
  toastId,
  onApplyNow,
  onLater,
  errorMessage,
}: ReadyToastParams): void => {
  toast.custom(
    () => (
      <UpdateToast
        version={version}
        onApplyNow={onApplyNow}
        onLater={onLater}
        errorMessage={errorMessage}
        onClose={() => dismissUpdateToast(toastId)}
      />
    ),
    {
      id: toastId,
      position: toastPosition,
      duration: Infinity,
    },
  );
};
