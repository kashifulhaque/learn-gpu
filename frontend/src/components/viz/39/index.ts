import type { VizRegistry } from "../registry";
import BrentBound from "./brent-bound";
import Litmus from "./litmus";

const figures: VizRegistry = {
  "39-brent-bound": BrentBound,
  "39-litmus": Litmus,
};

export default figures;
