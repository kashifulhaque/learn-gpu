import type { VizRegistry } from "../registry";
import FusionBytes from "./fusion-bytes";
import GraphTimeline from "./graph-timeline";

// Figures for chapter 17, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "17-fusion-bytes": FusionBytes,
  "17-graph-timeline": GraphTimeline,
};

export default figures;
