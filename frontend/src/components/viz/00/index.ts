import type { VizRegistry } from "../registry";

// Figures for chapter 00, keyed by the name a ```viz fence uses. Every name
// starts with the chapter's number and a hyphen, for example "00-example".
//
// To add a figure: write `./example.tsx` with a default-exported component
// that takes no props, import it here, and add `"00-example": Example`. The
// registry in ../registry.ts finds this file by itself; nothing else needs to
// change. content/STYLE.md documents the convention.
//
// This chapter has no figures yet, so the map is empty.
const figures: VizRegistry = {};

export default figures;
