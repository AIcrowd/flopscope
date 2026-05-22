import {
  canonicalTupleUnderGroup,
  restrictStabilizerToPositions,
} from './outputOrbit.js';

function tupleKey(labels, tuple) {
  return labels.map((label) => tuple[label]).join('|');
}

function vPositionsFromLabels(labels, vLabels) {
  const positionByLabel = new Map(labels.map((label, position) => [label, position]));
  return vLabels.map((label) => positionByLabel.get(label));
}

function tupleArrayFromObject(tuple, labels) {
  return labels.map((label) => tuple[label]);
}

function enumerateTuples(labels, sizesByLabel) {
  const tuples = [];
  const current = {};

  function visit(idx) {
    if (idx === labels.length) {
      tuples.push({ ...current });
      return;
    }

    const label = labels[idx];
    const size = sizesByLabel[label];
    for (let value = 0; value < size; value++) {
      current[label] = value;
      visit(idx + 1);
    }
  }

  visit(0);
  return tuples;
}

function applyPiToTuple(tuple, labels, perm) {
  const next = {};
  for (let i = 0; i < labels.length; i++) {
    next[labels[perm.arr[i]]] = tuple[labels[i]];
  }
  return next;
}

function projectTuple(tuple, vLabels) {
  const projected = {};
  for (const label of vLabels) projected[label] = tuple[label];
  return projected;
}

/**
 * Brute-force orbit enumeration. Canonical for n ≤ ~6, used as ground-truth
 * for the per-component aggregation tests in
 * `symmetry-explorer.cost-narrative.test.mjs`. The displayed μ and α on the
 * page come from `aggregateComponentCosts` (per-component decomposition);
 * this function exists to verify that the decomposition agrees with global
 * enumeration.
 */
export function computeExactCostModel({ labels, vLabels, groupElements, dimensionN, sizes, numTerms }) {
  // Build per-label size map. Prefer explicit `sizes` (size-aware presets);
  // fall back to uniform `dimensionN`.
  const sizesByLabel = {};
  if (Array.isArray(sizes) && sizes.length === labels.length) {
    labels.forEach((label, i) => { sizesByLabel[label] = sizes[i]; });
  } else {
    labels.forEach((label) => { sizesByLabel[label] = dimensionN; });
  }
  const tuples = enumerateTuples(labels, sizesByLabel);
  const effectiveElements = groupElements.length > 0
    ? groupElements
    : [{ arr: Array.from({ length: labels.length }, (_, idx) => idx) }];
  const visiblePositions = vPositionsFromLabels(labels, vLabels);
  const hElements = restrictStabilizerToPositions(effectiveElements, visiblePositions);
  const seen = new Set();
  const orbitRows = [];
  let reductionCostExact = 0;

  for (const tuple of tuples) {
    const repKey = tupleKey(labels, tuple);
    if (seen.has(repKey)) continue;

    const orbitMap = new Map();
    for (const element of effectiveElements) {
      const moved = applyPiToTuple(tuple, labels, element);
      orbitMap.set(tupleKey(labels, moved), moved);
    }

    for (const key of orbitMap.keys()) seen.add(key);

    const outputs = new Map();
    for (const moved of orbitMap.values()) {
      const outTuple = projectTuple(moved, vLabels);
      const outArray = tupleArrayFromObject(outTuple, vLabels);
      const outKey = canonicalTupleUnderGroup(outArray, hElements);
      const current = outputs.get(outKey) ?? { outTuple, outKey, coeff: 0 };
      current.coeff += 1;
      outputs.set(outKey, current);
    }

    const row = {
      repTuple: tuple,
      orbitSize: orbitMap.size,
      orbitTuples: [...orbitMap.values()],
      outputs: [...outputs.values()],
      outputCount: outputs.size,
    };
    orbitRows.push(row);
    reductionCostExact += row.outputCount;
  }

  const orbitCount = orbitRows.length;
  // α/M model is FMA=2-textbook by construction (multiplies AND adds
  // counted separately). No fma multiplier needed.
  const evaluationCostExact = Math.max(numTerms - 1, 0) * orbitCount;

  return {
    orbitCount,
    evaluationCostExact,
    reductionCostExact,
    orbitRows,
  };
}

/**
 * Aggregate per-component costs into the global μ, α, and total the hero
 * displays.
 *
 * The hero formula is
 *   Total = (k - 1) · ∏_a M_a  +  ∏_a α_a  −  ∏_a O_a
 *
 * which holds whenever the components are independent (G = ∏_a G_a, X = ∏_a X_a).
 * `decomposeClassifyAndCount` produces such components and attaches
 * `comp.multiplication.count = M_a`, `comp.accumulation.count = α_a`, and
 * `comp.numOutputOrbits = O_a` (the orbit count of the visible-label tuples
 * under the output stabilizer of that component).
 *
 * The `− ∏O_a` term applies the same off-by-one correction the reduction
 * surface uses: the first cell of each output orbit is a free copy, only the
 * remaining accumulations cost. When there is no actual reduction
 * (`∏α == ∏O`), the α contribution collapses to zero, leaving only the
 * `(k - 1)·∏M` multiplication chain.
 *
 * Returns `null` if any component has a missing accumulation count (e.g. the
 * regime ladder fell through, which currently can only happen when bruteForce
 * is disabled).
 */
export function aggregateComponentCosts(components, numTerms) {
  if (!Array.isArray(components) || components.length === 0) {
    return { mu: 0, alpha: 0, mTotal: 0, outputOrbitProduct: 0, total: 0, perComponent: [] };
  }

  let mTotal = 1;
  let alpha = 1;
  let outputOrbitProduct = 1;
  const perComponent = [];

  for (const comp of components) {
    const M_a = comp.multiplication?.count;
    const alpha_a = comp.accumulation?.count;
    const O_a = comp.numOutputOrbits;
    if (M_a == null || alpha_a == null || O_a == null) return null;
    mTotal *= M_a;
    alpha *= alpha_a;
    outputOrbitProduct *= O_a;
    perComponent.push({
      labels: comp.labels,
      M_a,
      alpha_a,
      O_a,
      regimeId: comp.accumulation?.regimeId ?? null,
    });
  }

  // α/M model is FMA=2-textbook by construction (multiplies AND adds
  // counted separately). No fma multiplier needed.
  const mu = Math.max(numTerms - 1, 0) * mTotal;
  const total = mu + alpha - outputOrbitProduct;
  return { mu, alpha, mTotal, outputOrbitProduct, total, perComponent };
}
