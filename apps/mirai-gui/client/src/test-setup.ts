// jsdom has no ResizeObserver, and menus measure their button when they open.
globalThis.ResizeObserver ??= class {
  observe() {}
  unobserve() {}
  disconnect() {}
};
