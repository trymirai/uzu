import { Button } from "./ui/button";
import { useRouter } from "@tanstack/react-router";
import { ArrowLeft } from "lucide-react";
import { AppLogo } from "./app-logo";
import { ErrorIcon } from "./icons/error-icon";

export function ErrorPage() {
  const router = useRouter();

  const handleAction = () => {
    if (window.history.length > 1) router.history.back();
    else router.navigate({ to: "/" });
  };

  return (
    <div className="h-full min-h-0 min-w-[360px] w-full flex flex-col bg-background text-label-title">
      <div className="shrink-0 flex px-8 py-8 justify-center">
        <AppLogo />
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto overscroll-y-contain px-4">
        <div className="flex min-h-full flex-col items-center justify-center gap-10 py-4">
          <ErrorIcon color="#FF2020" />
          <div className="flex flex-col items-center justify-center gap-3 text-center">
            <h1 className="text-2xl font-bold">Page Not Found</h1>
            <p className="text-base text-label-muted">An unexpected error occurred</p>
          </div>
          <Button kind="primary" icon={<ArrowLeft size={16} />} onClick={handleAction}>
            Go back
          </Button>
        </div>
      </div>
    </div>
  );
}
