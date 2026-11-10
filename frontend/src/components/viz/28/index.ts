import type { VizRegistry } from "../registry";
import RooflineCounters from "./roofline-counters";
import WaveSawtooth from "./wave-sawtooth";

const figures: VizRegistry = {
  "28-roofline-counters": RooflineCounters,
  "28-wave-sawtooth": WaveSawtooth,
};

export default figures;
