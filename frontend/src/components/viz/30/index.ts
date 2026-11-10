import type { VizRegistry } from "../registry";
import BitonicNetwork from "./bitonic-network";
import RadixPass from "./radix-pass";

const figures: VizRegistry = {
  "30-radix-pass": RadixPass,
  "30-bitonic-network": BitonicNetwork,
};

export default figures;
