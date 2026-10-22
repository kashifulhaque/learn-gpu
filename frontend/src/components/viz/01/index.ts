import type { VizRegistry } from "../registry";
import ChipAnatomy from "./chip";
import PeaksRidge from "./ridge";

// Figures for chapter 01, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "01-chip-anatomy": ChipAnatomy,
  "01-peaks-ridge": PeaksRidge,
};

export default figures;
