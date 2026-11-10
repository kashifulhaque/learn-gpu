import type { VizRegistry } from "../registry";
import RoundingGrid from "./rounding-grid";
import ScaleOutliers from "./scale-outliers";

const figures: VizRegistry = {
  "33-rounding-grid": RoundingGrid,
  "33-scale-outliers": ScaleOutliers,
};

export default figures;
