import type { VizRegistry } from "../registry";
import AllreduceCost from "./allreduce-cost";
import RingSteps from "./ring-steps";

// Figures for chapter 21, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "21-ring-steps": RingSteps,
  "21-allreduce-cost": AllreduceCost,
};

export default figures;
