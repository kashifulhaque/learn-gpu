import type { VizRegistry } from "../registry";
import ClusterFit from "./cluster-fit";
import MulticastTraffic from "./multicast-traffic";

const figures: VizRegistry = {
  "34-cluster-fit": ClusterFit,
  "34-multicast-traffic": MulticastTraffic,
};

export default figures;
