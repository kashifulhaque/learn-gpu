import type { VizRegistry } from "../registry";
import FormatRounding from "./format-rounding";
import SummationOrder from "./summation-order";

// Figures for chapter 09, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "09-format-rounding": FormatRounding,
  "09-summation-order": SummationOrder,
};

export default figures;
