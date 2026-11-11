import type { VizRegistry } from "../registry";
import PairTraffic from "./pair-traffic";
import TmemMap from "./tmem-map";

const figures: VizRegistry = {
  "35-tmem-map": TmemMap,
  "35-pair-traffic": PairTraffic,
};

export default figures;
