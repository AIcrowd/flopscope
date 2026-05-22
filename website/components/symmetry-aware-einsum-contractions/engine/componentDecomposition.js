/**
 * Component Decomposition Engine
 *
 * Decomposes the full symmetry group G into independent label components
 * (connected components of the label-interaction graph). Shape + regime
 * classification is owned by shapeLayer.js and the regime ladder — this
 * module only produces the components themselves.
 */

import { Permutation, dimino, burnsideCount } from './permutation.js';
import { detectShape } from './shapeLayer.js';
import { computeAccumulation } from './accumulationCount.js';
import { restrictStabilizerToPositions } from './outputOrbit.js';

class UnionFind {
  constructor(n) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.rank = new Array(n).fill(0);
  }

  find(x) {
    if (this.parent[x] !== x) this.parent[x] = this.find(this.parent[x]);
    return this.parent[x];
  }

  union(x, y) {
    const rx = this.find(x);
    const ry = this.find(y);
    if (rx === ry) return;
    if (this.rank[rx] < this.rank[ry]) {
      this.parent[rx] = ry;
    } else if (this.rank[rx] > this.rank[ry]) {
      this.parent[ry] = rx;
    } else {
      this.parent[ry] = rx;
      this.rank[rx] += 1;
    }
  }
}

export function buildLabelInteractionGraph(allLabels, generators) {
  const n = allLabels.length;
  const uf = new UnionFind(n);
  // Each edge is a 3-tuple [fromIdx, toIdx, generatorIdx]. The generator index
  // lets the UI attribute an edge back to the specific σ that introduced it
  // (see LabelInteractionGraph's edge tooltips). Downstream consumers that
  // only destructure [a, b] still work — the third element is silently
  // dropped by array destructuring.
  const edges = [];

  generators.forEach((gen, generatorIdx) => {
    const movedByThisGen = [];
    for (let i = 0; i < n; i += 1) {
      if (gen.arr[i] !== i) {
        movedByThisGen.push(i);
        edges.push([i, gen.arr[i], generatorIdx]);
        uf.union(i, gen.arr[i]);
      }
    }

    for (let j = 1; j < movedByThisGen.length; j += 1) {
      uf.union(movedByThisGen[0], movedByThisGen[j]);
    }
  });

  const componentMap = new Map();
  for (let i = 0; i < n; i += 1) {
    const root = uf.find(i);
    if (!componentMap.has(root)) componentMap.set(root, []);
    componentMap.get(root).push(i);
  }

  const components = [...componentMap.values()].map((indices) => indices.sort((a, b) => a - b));
  return { edges, components };
}

function restrictPermutation(perm, indices) {
  const localIdx = new Map(indices.map((globalIdx, localPos) => [globalIdx, localPos]));
  const arr = indices.map((globalIdx) => {
    const target = perm.arr[globalIdx];
    return localIdx.get(target);
  });
  if (arr.some((value) => value === undefined)) return null;
  return new Permutation(arr);
}

function factorial(n) {
  if (n <= 1) return 1;
  let result = 1;
  for (let i = 2; i <= n; i += 1) result *= i;
  return result;
}

function classifyGroupName(labels, generators, elements) {
  const order = elements.length;
  const degree = labels.length;
  if (degree < 2 || order <= 1) return 'trivial';

  const movedSet = new Set();
  for (const el of elements) {
    for (let i = 0; i < el.arr.length; i += 1) {
      if (el.arr[i] !== i) movedSet.add(i);
    }
  }

  const movedIndices = [...movedSet].sort((a, b) => a - b);
  const movedLabels = movedIndices.map((i) => labels[i]);
  const effectiveDegree = movedLabels.length || degree;
  const labelSet = `{${movedLabels.length > 0 ? movedLabels.join(',') : labels.join(',')}}`;
  const supportIndices = movedIndices.length > 0 ? movedIndices : labels.map((_, i) => i);

  const supportElements = elements
    .map((el) => restrictPermutation(el, supportIndices))
    .filter((el) => el !== null);

  if (order === factorial(effectiveDegree)) {
    const hasTransposition = supportElements.some((el) => {
      const cycles = el.cyclicForm();
      return cycles.length === 1 && cycles[0].length === 2;
    });
    if (hasTransposition || effectiveDegree <= 2) return `S${effectiveDegree}${labelSet}`;
  }

  if (order === effectiveDegree && effectiveDegree >= 3) {
    const hasFullCycle = supportElements.some((el) => {
      const cycles = el.cyclicForm();
      return cycles.length === 1 && cycles[0].length === effectiveDegree;
    });
    if (hasFullCycle) return `C${effectiveDegree}${labelSet}`;
  }

  if (order === 2 * effectiveDegree && effectiveDegree >= 3) {
    const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));
    const lcm = (a, b) => (a * b) / gcd(a, b);
    const permOrder = (el) => {
      const cycles = el.cyclicForm();
      if (cycles.length === 0) return 1;
      return cycles.reduce((acc, cycle) => lcm(acc, cycle.length), 1);
    };
    const rotations = supportElements.filter((el) => permOrder(el) === effectiveDegree);
    const reflections = supportElements.filter((el) => permOrder(el) === 2 && !el.isIdentity);
    let isDihedral = false;
    outer: for (const rot of rotations) {
      for (const ref of reflections) {
        const conj = ref.compose(rot).compose(ref);
        if (conj.equals(rot.inverse())) {
          isDihedral = true;
          break outer;
        }
      }
    }
    if (isDihedral) return `D${effectiveDegree}${labelSet}`;
  }

  if (order === 2) {
    if (effectiveDegree > 2) {
      // Single generator with disjoint 2-cycles: one coupled order-2 element,
      // not S2 × S2 (which would have order 4). Cycle notation makes the
      // order unambiguous.
      const gen = generators[0];
      const cycles = gen?.cyclicForm() || [];
      if (cycles.length > 1 && cycles.every((cycle) => cycle.length === 2)) {
        const genText = cycles
          .map((cycle) => `(${cycle.map((i) => labels[i]).join(' ')})`)
          .join('');
        return `\u27e8${genText}\u27e9`;
      }
      return `Z2${labelSet}`;
    }
    if (effectiveDegree === 2) return `S2${labelSet}`;
  }

  const genStr = generators.map((g) => g.cycleNotation(labels)).join(', ');
  return `PermGroup\u27e8${genStr}\u27e9`;
}

export function decomposeAndClassify(allLabels, vLabels, wLabels, fullGenerators, fullElements) {
  const vSet = new Set(vLabels);
  const wSet = new Set(wLabels);
  const effectiveGenerators = fullGenerators && fullGenerators.length > 0 ? fullGenerators : [];

  const interactionGraph = buildLabelInteractionGraph(allLabels, effectiveGenerators);
  const { components: rawComponents } = interactionGraph;

  const components = rawComponents.map((indices) => {
    const labels = indices.map((i) => allLabels[i]);
    const va = labels.filter((label) => vSet.has(label));
    const wa = labels.filter((label) => wSet.has(label));

    const restrictedGens = [];
    for (const gen of effectiveGenerators) {
      const restricted = restrictPermutation(gen, indices);
      if (restricted && !restricted.isIdentity) restrictedGens.push(restricted);
    }

    const seenKeys = new Set();
    const dedupGens = restrictedGens.filter((g) => {
      const key = g.key();
      if (seenKeys.has(key)) return false;
      seenKeys.add(key);
      return true;
    });

    const elements = dedupGens.length > 0 ? dimino(dedupGens) : [Permutation.identity(indices.length)];
    const order = elements.length;
    const groupName = classifyGroupName(labels, dedupGens, elements);

    return {
      indices,
      labels,
      va,
      wa,
      generators: dedupGens,
      elements,
      order,
      groupName,
    };
  });

  return { interactionGraph, components };
}

/**
 * Like decomposeAndClassify, but also attaches { shape, accumulation } to each
 * component using the shape layer + regime ladder from accumulationCount.js.
 *
 * @param {string[]} allLabels
 * @param {string[]} vLabels
 * @param {string[]} wLabels
 * @param {Permutation[]} fullGenerators
 * @param {Permutation[]} fullElements
 * @param {number[]} sizes - one entry per label in allLabels
 */
export function decomposeClassifyAndCount(
  allLabels, vLabels, wLabels, fullGenerators, fullElements, sizes,
) {
  const base = decomposeAndClassify(allLabels, vLabels, wLabels, fullGenerators, fullElements);
  const components = base.components.map((comp) => {
    const compSizes = comp.indices.map((i) => sizes[i]);
    const localLabels = comp.labels;
    const vPositionsLocal = comp.va.map((l) => localLabels.indexOf(l));
    const shape = detectShape({ va: comp.va, wa: comp.wa, elements: comp.elements });
    const accumulation = computeAccumulation({
      labels: localLabels,
      va: comp.va,
      wa: comp.wa,
      elements: comp.elements,
      sizes: compSizes,
      visiblePositions: vPositionsLocal,
      generators: comp.generators,
    });
    // Per-component orbit count M_a = Burnside on this component's group action.
    // For trivial components (identity-only) this equals ∏ compSizes.
    const multiplication = {
      count: burnsideCount(comp.elements, compSizes).uniqueCount,
    };
    // numOutputOrbits: orbits of the visible-label tuples under the output
    // stabilizer (subgroup of G that preserves V setwise, restricted to V).
    // Used by aggregateComponentCosts to apply the "first cell of each output
    // orbit is a free copy" off-by-one correction. Mirrors the Python
    // ComponentCost.num_output_orbits field. For scalar output (empty V),
    // there is one orbit; for trivial groups, this is just ∏ visible sizes.
    const vSizes = vPositionsLocal.map((p) => compSizes[p]);
    let numOutputOrbits;
    if (vSizes.length === 0) {
      numOutputOrbits = 1;
    } else if (comp.elements.length > 0) {
      const hElements = restrictStabilizerToPositions(comp.elements, vPositionsLocal);
      numOutputOrbits = burnsideCount(hElements, vSizes).uniqueCount;
    } else {
      numOutputOrbits = vSizes.reduce((a, b) => a * b, 1);
    }
    return {
      ...comp,
      sizes: compSizes,
      shape: shape.kind,
      accumulation,
      multiplication,
      numOutputOrbits,
    };
  });
  return { ...base, components };
}
