import type { VizRegistry } from "../registry";
import LayoutGrid from "./layout-grid";
import SwizzleBanks from "./swizzle-banks";

// Figures for chapter 13, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "13-layout-grid": LayoutGrid,
  "13-swizzle-banks": SwizzleBanks,
};

export default figures;
