import type { VizRegistry } from "../registry";
import Roofline from "./roofline";
import Waves from "./waves";

// Figures for chapter 04, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "04-roofline": Roofline,
  "04-waves": Waves,
};

export default figures;
