import type { VizRegistry } from "../registry";
import GroupScales from "./group-scales";
import W4a16Roofline from "./w4a16-roofline";

// Figures for chapter 18, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "18-group-scales": GroupScales,
  "18-w4a16-roofline": W4a16Roofline,
};

export default figures;
