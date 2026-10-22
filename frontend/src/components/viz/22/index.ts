import type { VizRegistry } from "../registry";
import PipelineBubble from "./pipeline-bubble";
import StreamTimeline from "./stream-timeline";

// Figures for chapter 22, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "22-stream-timeline": StreamTimeline,
  "22-pipeline-bubble": PipelineBubble,
};

export default figures;
