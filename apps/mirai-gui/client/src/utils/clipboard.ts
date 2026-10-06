export async function writeItemsWithFocus(items: ClipboardItem[]): Promise<void> {
  if (!document.hasFocus()) {
    window.focus();
  }
  await navigator.clipboard.write(items);
}
