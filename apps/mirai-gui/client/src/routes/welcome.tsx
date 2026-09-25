import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import WelcomeScreen from "../components/welcome-screen";

export const Route = createFileRoute("/welcome")({
  component: WelcomePage,
});

const waitFrames = (count: number): Promise<void> =>
  new Promise<void>((resolve) => {
    const step = (n: number): void => {
      if (n <= 0) {
        resolve();
        return;
      }
      requestAnimationFrame(() => step(n - 1));
    };
    step(count);
  });

function WelcomePage() {
  const [show, setShow] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void document.fonts.ready
      .catch(() => undefined)
      .then(() => waitFrames(10))
      .then(() => {
        if (!cancelled) setShow(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="relative w-screen h-screen flex items-start justify-center bg-bg dark:bg-[linear-gradient(180deg,#1A1A1A_0%,#0A0A0A_100%)] ">
      {show ? <WelcomeScreen /> : null}
    </div>
  );
}
