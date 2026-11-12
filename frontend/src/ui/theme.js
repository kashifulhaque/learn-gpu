// Generated from weights-and-wires/sites ui/ at b3af704 by scripts/sync-ui.sh. Do not edit here.

/*
 * Theme boot and toggle. Load it as a blocking <script> in <head>, before the
 * stylesheets, so the palette is set before first paint.
 *
 * With no stored choice the page follows the system setting. A click on any
 * element with [data-theme-toggle] flips the theme and stores the choice.
 * Storage can be blocked (private windows, strict settings); the toggle then
 * still works for the visit.
 *
 * Set data-theme-key on the <script> tag to use another storage key, for
 * example a course app that already stores its theme as "ml.theme".
 */
(function () {
  var script = document.currentScript;
  var key = (script && script.getAttribute("data-theme-key")) || "ifkash.theme";
  var root = document.documentElement;
  var media = window.matchMedia("(prefers-color-scheme: dark)");
  var stored = null;

  try {
    var value = localStorage.getItem(key);
    if (value === "light" || value === "dark") stored = value;
  } catch (e) {
    // Storage is unavailable. The system setting decides.
  }

  function current() {
    return stored || (media.matches ? "dark" : "light");
  }

  function apply() {
    var theme = current();
    root.setAttribute("data-theme", theme);
    root.style.colorScheme = theme;
    var meta = document.querySelector('meta[name="theme-color"]:not([media])');
    if (meta) meta.setAttribute("content", theme === "dark" ? "#15241e" : "#f7f6f0");
    var label = "Switch to " + (theme === "dark" ? "light" : "dark") + " theme";
    var toggles = document.querySelectorAll("[data-theme-toggle]");
    for (var i = 0; i < toggles.length; i++) {
      toggles[i].setAttribute("aria-label", label);
      toggles[i].setAttribute("title", label);
    }
  }

  function toggle() {
    stored = current() === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(key, stored);
    } catch (e) {
      // Keep the choice for this visit only.
    }
    apply();
  }

  apply();
  if (media.addEventListener) {
    media.addEventListener("change", function () {
      if (stored === null) apply();
    });
  }
  document.addEventListener("DOMContentLoaded", apply);
  document.addEventListener("click", function (event) {
    var target = event.target;
    if (target && target.closest && target.closest("[data-theme-toggle]")) toggle();
  });

  window.ifkashTheme = { current: current, toggle: toggle };
})();
