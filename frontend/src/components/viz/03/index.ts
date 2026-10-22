import type { VizRegistry } from "../registry";
import BytesInFlight from "./bytes-in-flight";
import Coalescing from "./coalescing";

// Figures for chapter 03, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "03-coalescing": Coalescing,
  "03-bytes-in-flight": BytesInFlight,
};

export default figures;
