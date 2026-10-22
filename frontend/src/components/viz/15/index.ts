import type { VizRegistry } from "../registry";
import OnlineSoftmax from "./online-softmax";
import RowPerProgram from "./row-per-program";

// Figures for chapter 15, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "15-online-softmax": OnlineSoftmax,
  "15-row-per-program": RowPerProgram,
};

export default figures;
