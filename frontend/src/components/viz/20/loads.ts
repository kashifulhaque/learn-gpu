// Shared by the chapter 20 figures. Expert loads are a model, not a measurement:
// the expected share of expert e is proportional to 1 / (e + 1)^skew, so skew 0 is
// perfectly balanced and a larger skew makes low-numbered experts popular.

/** Split `rows` across `experts` with weights 1 / (e + 1)^skew, by largest remainder. */
export function expertLoads(rows: number, experts: number, skew: number): number[] {
  const weights = Array.from({ length: experts }, (_, e) => Math.pow(1 / (e + 1), skew));
  const total = weights.reduce((a, b) => a + b, 0);
  const exact = weights.map((w) => (rows * w) / total);
  const counts = exact.map(Math.floor);
  let left = rows - counts.reduce((a, b) => a + b, 0);
  const order = exact.map((v, i) => [v - Math.floor(v), i] as const).sort((p, q) => q[0] - p[0]);
  for (const [, i] of order) {
    if (left <= 0) break;
    counts[i] += 1;
    left -= 1;
  }
  return counts;
}

/** The tiles of `blockM` rows that a grouped GEMM launches: ceil(count / blockM) per expert. */
export function tileCount(counts: readonly number[], blockM: number): number {
  return counts.reduce((a, c) => a + Math.ceil(c / blockM), 0);
}
