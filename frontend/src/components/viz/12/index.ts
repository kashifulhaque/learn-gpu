import type { VizRegistry } from "../registry";
import PipelineTimeline from "./pipeline-timeline";
import RingBuffer from "./ring-buffer";

// Figures for chapter 12, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "12-pipeline-timeline": PipelineTimeline,
  "12-ring-buffer": RingBuffer,
};

export default figures;
