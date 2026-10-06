import { autoUpdate } from "@floating-ui/react-dom";
import type { ReferenceType } from "@floating-ui/react-dom";

export function autoUpdateWithDetach(
  reference: ReferenceType,
  floating: HTMLElement,
  update: () => void,
  onDetach: () => void,
) {
  return autoUpdate(
    reference,
    floating,
    () => {
      if (reference instanceof Element && !reference.isConnected) {
        onDetach();
        return;
      }
      update();
    },
    { animationFrame: true },
  );
}

function getDescribedByValue(currentValue: string | null, tooltipId: string, includeTooltipId: boolean) {
  const tokens = (currentValue ?? "")
    .split(" ")
    .map((token) => token.trim())
    .filter((token) => token !== "" && token !== tooltipId);

  if (includeTooltipId) {
    tokens.push(tooltipId);
  }

  return tokens.join(" ") || null;
}

export function syncAriaDescribedBy(element: HTMLElement | null, tooltipId: string, includeTooltipId: boolean) {
  if (!element) return;

  const nextValue = getDescribedByValue(element.getAttribute("aria-describedby"), tooltipId, includeTooltipId);

  if (nextValue) {
    element.setAttribute("aria-describedby", nextValue);
    return;
  }

  element.removeAttribute("aria-describedby");
}
