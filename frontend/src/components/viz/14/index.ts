import type { VizRegistry } from "../registry";
import GroupedOrdering from "./grouped-ordering";
import TileBudget from "./tile-budget";

// Figures for chapter 14, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "14-grouped-ordering": GroupedOrdering,
  "14-tile-budget": TileBudget,
};

export default figures;
