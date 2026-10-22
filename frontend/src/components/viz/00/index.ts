import type { VizRegistry } from "../registry";
import AsyncTimeline from "./async-timeline";
import ThreadsInFlight from "./threads-in-flight";

// Figures for chapter 00, keyed by the name a ```viz fence uses.
const figures: VizRegistry = {
  "00-async-timeline": AsyncTimeline,
  "00-threads-in-flight": ThreadsInFlight,
};

export default figures;
