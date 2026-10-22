import type { VizRegistry } from "../registry";
import ChunkedXent from "./chunked-xent";
import LogitsMemory from "./logits-memory";

// Figures for chapter 19, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "19-logits-memory": LogitsMemory,
  "19-chunked-xent": ChunkedXent,
};

export default figures;
