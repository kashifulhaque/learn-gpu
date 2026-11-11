import type { VizRegistry } from "../registry";
import ChunkTradeoff from "./chunk-tradeoff";
import DecayUnderflow from "./decay-underflow";

const figures: VizRegistry = {
  "37-chunk-tradeoff": ChunkTradeoff,
  "37-decay-underflow": DecayUnderflow,
};

export default figures;
