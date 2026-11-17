import type { SystemUiService } from ".";
import { noopUnsubscribe } from "../shared/noop";

export const webSystemUi: SystemUiService = {
  setWindowTheme: () => Promise.resolve(false),
  getZoom: () => Promise.resolve(1),
  stepZoom: () => Promise.resolve(1),
  onZoomChange: noopUnsubscribe,
};
