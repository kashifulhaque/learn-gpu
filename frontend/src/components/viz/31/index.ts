import type { VizRegistry } from "../registry";
import HaloOverhead from "./halo-overhead";
import Im2colBlowup from "./im2col-blowup";

const figures: VizRegistry = {
  "31-halo-overhead": HaloOverhead,
  "31-im2col-blowup": Im2colBlowup,
};

export default figures;
