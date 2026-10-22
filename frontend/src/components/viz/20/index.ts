import type { VizRegistry } from "../registry";
import RoutingLoad from "./routing-load";
import TileScheduler from "./tile-scheduler";

// Figures for chapter 20, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "20-routing-load": RoutingLoad,
  "20-tile-scheduler": TileScheduler,
};

export default figures;
