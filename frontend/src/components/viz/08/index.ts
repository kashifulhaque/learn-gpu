import type { VizRegistry } from "../registry";
import BankMap from "./bank-map";
import TilePath from "./tile-path";

// Figures for chapter 08, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "08-bank-map": BankMap,
  "08-tile-path": TilePath,
};

export default figures;
