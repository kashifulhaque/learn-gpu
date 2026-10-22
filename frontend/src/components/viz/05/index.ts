import type { VizRegistry } from "../registry";
import LaunchGaps from "./launch-gaps";
import WarpDivergence from "./warp-divergence";

// Figures for chapter 05, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "05-launch-gaps": LaunchGaps,
  "05-warp-divergence": WarpDivergence,
};

export default figures;
