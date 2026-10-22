import type { VizRegistry } from "../registry";
import FragmentMap from "./fragment-map";
import LdmatrixBanks from "./ldmatrix-banks";

// Figures for chapter 11, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "11-fragment-map": FragmentMap,
  "11-ldmatrix-banks": LdmatrixBanks,
};

export default figures;
