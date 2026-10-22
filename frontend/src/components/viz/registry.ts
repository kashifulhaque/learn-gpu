// Every chapter figure, by the name a ```viz fence uses.
//
// Each chapter keeps its figures in its own folder, `viz/<NN>/`, where NN is
// the chapter's two-digit number, and that folder's `index.ts` default-exports
// a name-to-component map. This file collects every such map with a glob, so
// two people writing figures for different chapters never edit the same file.
// A figure's name starts with its chapter's number, for example `03-coalescing`;
// scripts/check_chapters.py reads the same keys to catch a fence with a typo.

import type { ComponentType } from "react";

export type VizRegistry = Record<string, ComponentType>;

const groups = import.meta.glob<VizRegistry>("./[0-9][0-9]/index.ts", {
  eager: true,
  import: "default",
});

function collect(): VizRegistry {
  const all: VizRegistry = {};
  for (const [path, figures] of Object.entries(groups)) {
    const chapter = /\/(\d\d)\/index\.ts$/.exec(path)?.[1] ?? "";
    for (const [name, component] of Object.entries(figures ?? {})) {
      if (!name.startsWith(`${chapter}-`)) {
        console.error(`The figure "${name}" is registered in viz/${chapter}/ but does not start with "${chapter}-".`);
      }
      if (name in all) {
        console.error(`The figure name "${name}" is registered twice.`);
      }
      all[name] = component;
    }
  }
  return all;
}

export const VIZ: VizRegistry = collect();
