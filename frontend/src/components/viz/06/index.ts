import type { VizRegistry } from "../registry";
import TreeRounds from "./tree";
import WarpShuffle from "./shuffle";

// Figures for chapter 06, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "06-tree-rounds": TreeRounds,
  "06-warp-shuffle": WarpShuffle,
};

export default figures;
