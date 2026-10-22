import type { VizRegistry } from "../registry";
import TileIntensity from "./tile-intensity";
import OuterProduct from "./outer-product";

// Figures for chapter 10, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "10-tile-intensity": TileIntensity,
  "10-outer-product": OuterProduct,
};

export default figures;
