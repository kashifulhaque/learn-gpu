import type { VizRegistry } from "../registry";
import AllocatorSteps from "./allocator-steps";
import BudgetBar from "./budget-bar";

const figures: VizRegistry = {
  "40-allocator-steps": AllocatorSteps,
  "40-budget-bar": BudgetBar,
};

export default figures;
