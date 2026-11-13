// Whether this device has been through the welcome. A per-device first-run flag, so local storage is
// enough; if it can't be read the welcome is simply shown again, which is harmless.
const KEY = "mirai.welcome-seen";

export function hasSeenWelcome(): boolean {
  try {
    return localStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}

export function markWelcomeSeen(): void {
  try {
    localStorage.setItem(KEY, "1");
  } catch {
    // Nothing to do: the welcome shows once more next launch.
  }
}
