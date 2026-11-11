import type { VizRegistry } from "../registry";
import BfsDirections from "./bfs-directions";
import WarpWork from "./warp-work";

const figures: VizRegistry = {
  "32-warp-work": WarpWork,
  "32-bfs-directions": BfsDirections,
};

export default figures;
