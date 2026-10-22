import type { VizRegistry } from "../registry";
import HistogramContention from "./histogram-contention";
import ScanNetworks from "./scan-networks";

// Figures for chapter 07, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "07-scan-networks": ScanNetworks,
  "07-histogram-contention": HistogramContention,
};

export default figures;
