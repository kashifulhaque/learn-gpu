import type { VizRegistry } from "../registry";
import Divergence from "./divergence";
import OccupancyCalculator from "./occupancy-calculator";

// Figures for chapter 02, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "02-occupancy-calculator": OccupancyCalculator,
  "02-divergence": Divergence,
};

export default figures;
