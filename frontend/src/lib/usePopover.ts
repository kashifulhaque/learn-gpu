import { useCallback, useEffect, useId, useRef, useState } from "react";

/**
 * Open and close state for a button-and-popover pair. A click outside the
 * pair or Escape closes the popover, and Escape returns focus to the button.
 *
 * Spread `buttonProps` on the trigger and `panelProps` on the popover, and
 * wrap both in an element that takes `rootRef`.
 *
 * Portable: no course-specific imports.
 */
export function usePopover<T extends HTMLElement = HTMLDivElement>() {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<T>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const id = useId();

  const close = useCallback((returnFocus = false) => {
    setOpen(false);
    if (returnFocus) buttonRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!open) return;
    function onPointer(event: PointerEvent) {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    }
    function onKey(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      close(true);
    }
    window.addEventListener("pointerdown", onPointer);
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("pointerdown", onPointer);
      window.removeEventListener("keydown", onKey, true);
    };
  }, [open, close]);

  return {
    open,
    setOpen,
    close,
    rootRef,
    buttonProps: {
      ref: buttonRef,
      type: "button" as const,
      "aria-expanded": open,
      "aria-controls": id,
      onClick: () => setOpen((value) => !value),
    },
    panelProps: { id },
  };
}
