import type { VizRegistry } from "../registry";
import BlockedLayout from "./blocked-layout";
import PipelineSchedule from "./pipeline-schedule";

const figures: VizRegistry = {
  "38-blocked-layout": BlockedLayout,
  "38-pipeline-schedule": PipelineSchedule,
};

export default figures;
