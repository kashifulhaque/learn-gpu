import type { VizRegistry } from "../registry";
import AcceleratorRidge from "./accelerator-ridge";
import WavefrontWidth from "./wavefront-width";

// Figures for chapter 23, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "23-accelerator-ridge": AcceleratorRidge,
  "23-wavefront-width": WavefrontWidth,
};

export default figures;
