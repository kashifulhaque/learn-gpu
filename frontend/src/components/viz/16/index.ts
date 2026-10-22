import type { VizRegistry } from "../registry";
import IoTraffic from "./io-traffic";
import TileGrid from "./tile-grid";

// Figures for chapter 16, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "16-io-traffic": IoTraffic,
  "16-tile-grid": TileGrid,
};

export default figures;
