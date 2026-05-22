/**
 * Core algorithm engine — JS port of the subgraph symmetry detection.
 * Ports logic from flopscope/_opt_einsum/_subgraph_symmetry.py
 */

import { Permutation, dimino, burnsideCount } from './permutation.js';
import { buildFullGroup, classifyPi } from './fullGroup.js';
import { enumerateWreath } from './wreath.js';

// ─── Bipartite Graph Construction ─────────────────────────────────

/**
 * Build the bipartite graph for an einsum expression.
 * Port of _build_bipartite (subgraph_symmetry.py:174-277)
 */
export function buildBipartite(example) {
  const { subscripts, output, operandNames } = example;
  const numOps = subscripts.length;

  const uVertices = [];  // { opIdx, classId, labels: Set<string> }
  const incidence = [];  // { [label]: multiplicity }
  const uOperand = [];
  const operandLabels = [];

  for (let opIdx = 0; opIdx < numOps; opIdx++) {
    const sub = subscripts[opIdx];
    operandLabels.push(new Set(sub));

    // Each axis gets its own class (one U-vertex per axis).
    // No axis merging — per-operand symmetry handled by σ-loop generators.
    const classOf = {};
    for (let k = 0; k < sub.length; k++) classOf[k] = k;

    // Normalize class ids
    const classOrder = {};
    let nextClass = 0;
    for (let k = 0; k < sub.length; k++) {
      const c = classOf[k];
      if (!(c in classOrder)) classOrder[c] = nextClass++;
      classOf[k] = classOrder[c];
    }

    // Build one U-vertex per class
    const numClasses = nextClass;
    const classInc = Array.from({ length: numClasses }, () => ({}));
    const classLabels = Array.from({ length: numClasses }, () => new Set());

    for (let k = 0; k < sub.length; k++) {
      const cls = classOf[k];
      const ch = sub[k];
      classInc[cls][ch] = (classInc[cls][ch] || 0) + 1;
      classLabels[cls].add(ch);
    }

    for (let cls = 0; cls < numClasses; cls++) {
      uVertices.push({ opIdx, classId: cls, labels: classLabels[cls] });
      incidence.push(classInc[cls]);
      uOperand.push(opIdx);
    }
  }

  // All labels
  const allLabels = new Set();
  operandLabels.forEach(s => s.forEach(l => allLabels.add(l)));
  const outputSet = new Set(output);
  const freeLabels = new Set([...allLabels].filter(l => outputSet.has(l)));
  const summedLabels = new Set([...allLabels].filter(l => !outputSet.has(l)));

  // Identical-operand groups (by name — simulates Python `is`)
  const nameToPositions = {};
  for (let i = 0; i < numOps; i++) {
    const name = operandNames[i];
    (nameToPositions[name] ??= []).push(i);
  }
  const identicalGroups = Object.values(nameToPositions).filter(g => g.length >= 2);

  return {
    uVertices, incidence, uOperand, operandLabels,
    freeLabels, summedLabels, identicalGroups,
    allLabels: [...allLabels].sort(),
    numOperands: numOps,
  };
}

// ─── Incidence Matrix ─────────────────────────────────────────────

export function buildIncidenceMatrix(graph) {
  const { uVertices, incidence, allLabels } = graph;
  const labels = allLabels;
  const matrix = uVertices.map((_, rowIdx) =>
    labels.map(lbl => incidence[rowIdx][lbl] || 0)
  );

  // Column fingerprints
  const colFingerprints = {};
  const fpToLabels = {};
  for (let c = 0; c < labels.length; c++) {
    const fp = matrix.map(row => row[c]).join(',');
    colFingerprints[labels[c]] = fp;
    (fpToLabels[fp] ??= new Set()).add(labels[c]);
  }

  // Row labels for display
  const rowLabels = uVertices.map(u => {
    return { opIdx: u.opIdx, classId: u.classId, labels: u.labels };
  });

  return { matrix, labels, rowLabels, colFingerprints, fpToLabels };
}

// ─── σ-Loop: enumerate operand permutations ───────────────────────

/**
 * Derive π from σ(M) via column fingerprint matching.
 * Port of _derive_pi_canonical (subgraph_symmetry.py:32-70)
 */
function derivePi(sigmaColOf, fpToLabels, vLabels, wLabels) {
  const pi = {};
  const used = new Set();
  const allLabels = [...vLabels, ...wLabels].sort();

  // In the full-group model, π may legitimately mix V and W labels.
  // Cross V/W actions are part of the detected symmetry, so we only require
  // a bijective fingerprint match here and do not reintroduce the old
  // partition-preserving rejection.
  for (const label of allLabels) {
    const fp = sigmaColOf[label];
    const candidates = fpToLabels[fp];
    if (!candidates) return null;
    let pick = null;
    for (const c of [...candidates].sort()) {
      if (!used.has(c)) { pick = c; break; }
    }
    if (pick === null) return null;
    pi[label] = pick;
    used.add(pick);
  }

  return pi;
}

/**
 * Run the full σ-loop. Returns results for each wreath-product element.
 *
 * Directly enumerates the wreath product ∏_i (H_i ≀ S_{m_i}) via
 * enumerateWreath, replacing the old Source A/B generator emission +
 * Dimino-#1 closure pipeline. Per AUDIT Appendix A, the result is an
 * identity-equivalent set.
 *
 * @returns {{results: Array, wreathElements: Array}} results is the Shape-1/2/3
 *   schema consumed by buildGroup/SigmaLoop/DiminoView (unchanged from Task 3).
 *   wreathElements is a parallel array with 3-way classification entries aligned
 *   index-by-index with results.
 */
export function runSigmaLoop(graph, matrixData, example) {
  const { freeLabels, summedLabels, allLabels, incidence, uOperand } = graph;
  const { fpToLabels } = matrixData;

  const vLabels = freeLabels;
  const wLabels = summedLabels;
  const rowOrder = Array.from({ length: graph.uVertices.length }, (_, i) => i);

  // Build opToUIndices: operand index → list of row positions in rowOrder
  const opToUIndices = {};
  for (let k = 0; k < rowOrder.length; k++) {
    const opIdx = uOperand[rowOrder[k]];
    (opToUIndices[opIdx] ??= []).push(k);
  }

  const results = [];
  const wreathElements = [];

  // ── Directly enumerate the wreath product ∏_i (H_i ≀ S_{m_i}) ──
  // (Replaces the old Source A/B emitters + Dimino-#1 closure.
  // Per AUDIT Appendix A, the result is an identity-equivalent set.)
  const axisRanks = example.subscripts.map((s) => s.length);
  const perOpSymmetry = example?.perOpSymmetry;
  // Pass per-operand symmetry aligned by operand position. If perOpSymmetry
  // is a single object (legacy shape), broadcast it to all operands so
  // enumerateH sees the right shape.
  const perOpSymmetryArr = Array.isArray(perOpSymmetry)
    ? perOpSymmetry
    : axisRanks.map(() => perOpSymmetry ?? null);

  // Compute uOffsets from opToUIndices so enumerateWreath's row-perm
  // coordinate frame matches graph.uVertices indexing.
  const nOperands = axisRanks.length;
  const uOffsets = new Array(nOperands);
  for (let p = 0; p < nOperands; p += 1) {
    const positions = opToUIndices[p] || [];
    uOffsets[p] = positions.length > 0 ? positions[0] : 0;
  }

  // identicalGroups contains only groups of size >= 2. For operands with no
  // partner we add singleton groups so the wreath enumerator walks every
  // operand's H_i.
  const groupedOperands = new Set();
  for (const g of graph.identicalGroups) for (const p of g) groupedOperands.add(p);
  const singletonGroups = [];
  for (let p = 0; p < nOperands; p += 1) {
    if (!groupedOperands.has(p)) singletonGroups.push([p]);
  }
  const identicalGroupsAll = [...graph.identicalGroups, ...singletonGroups];

  // (Source C removed: it emitted expression-level dummy-rename symmetries
  // on summed axes of identical operands. Those are invariant under the
  // total sum but NOT at the per-tuple level required by Burnside orbit
  // compression, so feeding them to the compression pipeline caused
  // over-compression on Frobenius-style einsums. Expression-level
  // symmetries are now recovered separately via G_expr = V-sub × S(W)
  // for pedagogical display without contaminating compression.)

  for (const element of enumerateWreath({
    identicalGroups: identicalGroupsAll,
    perOpSymmetry: perOpSymmetryArr,
    axisRanks,
    uOffsets,
  })) {
    const sigmaRowPerm = element.rowPerm.arr;
    const isIdentity = sigmaRowPerm.every((v, i) => v === i);

    if (isIdentity) {
      const identityPi = Object.fromEntries(allLabels.map((label) => [label, label]));
      results.push({
        sigma: {},
        isIdentity: true,
        skipped: true,
        pi: identityPi,
        isValid: true,
        piIsIdentity: true,
        piKind: 'identity',
        piCrossesVw: false,
        piMovesV: false,
        piMovesW: false,
      });
      wreathElements.push({
        id: `w${wreathElements.length}`,
        rowPerm: element.rowPerm,
        derivePiResult: identityPi,
        matrixPreserving: true,
        isValid: true,
        classification: 'matrix-preserving',
        factorization: element.factorization,
      });
      continue;
    }

    // Compute σ(M) column fingerprints
    const sigmaColOf = {};
    for (const label of allLabels) {
      sigmaColOf[label] = rowOrder.map(
        (_, k) => incidence[sigmaRowPerm[k]][label] || 0
      ).join(',');
    }

    // Build σ(M) matrix for display
    const sigmaMatrix = sigmaRowPerm.map(uIdx =>
      allLabels.map(lbl => incidence[uIdx][lbl] || 0)
    );

    // Derive π
    const pi = derivePi(sigmaColOf, fpToLabels, vLabels, wLabels);
    if (!pi) {
      results.push({
        sigma: {}, sigmaRowPerm, sigmaMatrix, sigmaColOf,
        isValid: false, reason: 'No matching π (fingerprint mismatch)',
      });
      wreathElements.push({
        id: `w${wreathElements.length}`,
        rowPerm: element.rowPerm,
        derivePiResult: null,
        matrixPreserving: false,
        isValid: false,
        classification: 'rejected',
        factorization: element.factorization,
      });
      continue;
    }

    const {
      piIsIdentity,
      piKind,
      crosses: piCrossesVw,
      movesV: piMovesV,
      movesW: piMovesW,
    } = classifyPi(pi, vLabels, wLabels);

    results.push({
      sigma: {}, sigmaRowPerm, sigmaMatrix, sigmaColOf, pi,
      isValid: true, piIsIdentity, piKind,
      piCrossesVw, piMovesV, piMovesW,
    });
    wreathElements.push({
      id: `w${wreathElements.length}`,
      rowPerm: element.rowPerm,
      derivePiResult: pi,
      matrixPreserving: piIsIdentity,
      isValid: true,
      classification: piIsIdentity ? 'matrix-preserving' : 'valid',
      factorization: element.factorization,
    });
  }

  return { results, wreathElements };
}

// ─── Group Construction ───────────────────────────────────────────

/**
 * Reduce a set of generators to a minimal generating set.
 * Greedily adds generators that strictly grow the group.
 */
function minimalGenerators(gens) {
  if (gens.length <= 1) return gens;
  const minimal = [];
  let currentSize = 1; // just identity
  for (const g of gens) {
    const trial = [...minimal, g];
    const elems = dimino(trial);
    if (elems.length > currentSize) {
      minimal.push(g);
      currentSize = elems.length;
    }
  }
  return minimal;
}

export function buildGroup(sigmaResults, graph) {
  const allLabels = [...graph.allLabels].sort();
  const vLabels = [...graph.freeLabels].sort();
  const wLabels = [...graph.summedLabels].sort();

  const vIdx = {};
  vLabels.forEach((l, i) => vIdx[l] = i);
  const wIdx = {};
  wLabels.forEach((l, i) => wIdx[l] = i);

  const vGenerators = [];
  const wGenerators = [];

  for (const r of sigmaResults) {
    if (!r.isValid || r.skipped || r.piIsIdentity) continue;
    if (r.piKind !== 'v-only' && r.piKind !== 'w-only') continue;
    const { pi } = r;

    // Restrict π to V labels
    if (r.piKind === 'v-only' && vLabels.length >= 2) {
      const arr = vLabels.map((l) => {
        const target = pi[l];
        return vLabels.includes(target) ? vIdx[target] : vIdx[l];
      });
      const perm = new Permutation(arr);
      if (!perm.isIdentity) vGenerators.push(perm);
    }

    // Restrict π to W labels
    if (r.piKind === 'w-only' && wLabels.length >= 2) {
      const arr = wLabels.map((l) => {
        const target = pi[l];
        return wLabels.includes(target) ? wIdx[target] : wIdx[l];
      });
      const perm = new Permutation(arr);
      if (!perm.isIdentity) wGenerators.push(perm);
    }
  }

  const validPiResults = sigmaResults
    .filter((r) => r.isValid && r.pi)
    .map((r) => ({ ...r }));

  const fullGroup = buildFullGroup(allLabels, validPiResults, vLabels, wLabels);

  // Deduplicate generators
  const dedup = (gens) => {
    const seen = new Set();
    return gens.filter(g => {
      const k = g.key();
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
  };

  const vGensAll = dedup(vGenerators);
  const wGensAll = dedup(wGenerators);

  // Reduce to a minimal generating set: only keep generators that grow the group
  const vGens = minimalGenerators(vGensAll);
  const wGens = minimalGenerators(wGensAll);

  const vElements = vGens.length > 0 ? dimino(vGens) : (vLabels.length > 0 ? [Permutation.identity(vLabels.length)] : []);
  const wElements = wGens.length > 0 ? dimino(wGens) : (wLabels.length > 0 ? [Permutation.identity(wLabels.length)] : []);

  // Classify group — use Python PathInfo convention: S2{a,b}, C3{i,j,k}, etc.
  // When the group is embedded (acts on a subset of labels, fixing the rest),
  // use the effective degree (number of moved labels) for classification, and
  // show only the moved labels in the label set.
  const vOrder = vElements.length;
  const vDegree = vLabels.length;
  let vGroupName = 'trivial';
  if (vDegree >= 2 && vOrder > 1) {
    // Find which labels are actually moved by any group element
    const movedSet = new Set();
    for (const el of vElements) {
      for (let i = 0; i < el.arr.length; i++) {
        if (el.arr[i] !== i) movedSet.add(i);
      }
    }
    const movedLabels = [...movedSet].sort((a, b) => a - b).map(i => vLabels[i]);
    const effectiveDegree = movedLabels.length || vDegree;
    const factorial = (n) => n <= 1 ? 1 : n * factorial(n - 1);
    const labelSet = `{${movedLabels.length > 0 ? movedLabels.join(',') : vLabels.join(',')}}`;
    if (vOrder === factorial(effectiveDegree)) {
      vGroupName = `S${effectiveDegree}${labelSet}`;
    } else if (vOrder === effectiveDegree && effectiveDegree >= 3) {
      vGroupName = `C${effectiveDegree}${labelSet}`;
    } else if (vOrder === 2 * effectiveDegree && effectiveDegree >= 3) {
      vGroupName = `D${effectiveDegree}${labelSet}`;
    } else if (vOrder === 2 && effectiveDegree > 2) {
      const gen = vGens[0];
      const cycles = gen.cyclicForm();
      const allTwoCycles = cycles.every(c => c.length === 2);
      if (allTwoCycles && cycles.length > 1) {
        // Format as product of S2 on each orbit
        const orbitParts = cycles.map(c => `S2{${c.map(i => vLabels[i]).join(',')}}`);
        vGroupName = orbitParts.join('\u00d7');  // × multiplication sign
      } else {
        vGroupName = `Z2${labelSet}`;
      }
    } else if (vOrder === 2 && vDegree === 2) {
      vGroupName = `S2${labelSet}`;
    } else {
      // Fallback: show generators in cycle notation (matches Python PermGroup⟨...⟩)
      const genStr = vGens.map(g => g.cycleNotation(vLabels)).join(', ');
      vGroupName = `PermGroup\u27e8${genStr}\u27e9`;
    }
  }

  // Classify W-side group the same way
  const wOrder = wElements.length;
  const wDegree = wLabels.length;
  let wGroupName = 'trivial';
  if (wDegree >= 2 && wOrder > 1) {
    const wMovedSet = new Set();
    for (const el of wElements) {
      for (let i = 0; i < el.arr.length; i++) {
        if (el.arr[i] !== i) wMovedSet.add(i);
      }
    }
    const wMovedLabels = [...wMovedSet].sort((a, b) => a - b).map(i => wLabels[i]);
    const wEffectiveDegree = wMovedLabels.length || wDegree;
    const factorial = (n) => n <= 1 ? 1 : n * factorial(n - 1);
    const wLabelSet = `{${wMovedLabels.length > 0 ? wMovedLabels.join(',') : wLabels.join(',')}}`;
    if (wOrder === factorial(wEffectiveDegree)) {
      wGroupName = `S${wEffectiveDegree}${wLabelSet}`;
    } else if (wOrder === wEffectiveDegree && wEffectiveDegree >= 3) {
      wGroupName = `C${wEffectiveDegree}${wLabelSet}`;
    } else if (wOrder === 2 * wEffectiveDegree && wEffectiveDegree >= 3) {
      wGroupName = `D${wEffectiveDegree}${wLabelSet}`;
    } else if (wOrder === 2 && wEffectiveDegree > 2) {
      const gen = wGens[0];
      const cycles = gen.cyclicForm();
      const allTwoCycles = cycles.every(c => c.length === 2);
      if (allTwoCycles && cycles.length > 1) {
        const orbitParts = cycles.map(c => `S2{${c.map(i => wLabels[i]).join(',')}}`);
        wGroupName = orbitParts.join('\u00d7');
      } else {
        wGroupName = `Z2${wLabelSet}`;
      }
    } else if (wOrder === 2 && wDegree === 2) {
      wGroupName = `S2${wLabelSet}`;
    } else {
      const genStr = wGens.map(g => g.cycleNotation(wLabels)).join(', ');
      wGroupName = `PermGroup\u27e8${genStr}\u27e9`;
    }
  }

  return {
    allLabels,
    vLabels, wLabels,
    vGenerators: vGens, vGeneratorsAll: vGensAll,
    wGenerators: wGens,
    vElements, wElements,
    vOrder, vGroupName,
    wOrder, wGroupName,
    vDegree, wDegree,
    fullGenerators: fullGroup.fullGenerators,
    fullElements: fullGroup.fullElements,
    fullOrder: fullGroup.fullOrder,
    fullDegree: fullGroup.fullDegree,
    fullGroupName: fullGroup.fullGroupName,
    actionSummary: fullGroup.actionSummary,
    validPiResults: fullGroup.validPiResults,
    generatorSelection: fullGroup.generatorSelection,
  };
}

// ─── Declared Symmetry (user-provided, not detected) ────────────

/**
 * Build a group directly from a declared symmetry type.
 * Used for examples where the user declares symmetry on a single tensor
 * (not auto-detected from identical operands in an einsum).
 */
export function buildDeclaredGroup(example) {
  const labels = [...example.output].sort();
  const k = labels.length;
  const sym = Array.isArray(example.perOpSymmetry)
    ? example.perOpSymmetry[0]
    : example.perOpSymmetry;

  let vGens = [];

  if (sym === 'symmetric') {
    // S_k: adjacent transpositions
    for (let i = 0; i < k - 1; i++) {
      const arr = Array.from({ length: k }, (_, j) => j);
      arr[i] = i + 1;
      arr[i + 1] = i;
      vGens.push(new Permutation(arr));
    }
  } else if (sym === 'cyclic') {
    // C_k: single k-cycle (0 → 1 → 2 → ... → k-1 → 0)
    const arr = Array.from({ length: k }, (_, j) => (j + 1) % k);
    vGens.push(new Permutation(arr));
  } else if (sym === 'dihedral') {
    // D_k: rotation + reflection
    const rotation = Array.from({ length: k }, (_, j) => (j + 1) % k);
    vGens.push(new Permutation(rotation));
    if (k >= 3) {
      const reflection = [0, ...Array.from({ length: k - 1 }, (_, j) => k - 1 - j)];
      vGens.push(new Permutation(reflection));
    }
  } else if (sym === 'block-swap') {
    // Block swap: (0,1) ↔ (2,3) via (0 2)(1 3)
    if (k === 4) {
      vGens.push(new Permutation([2, 3, 0, 1]));
    }
  }

  if (vGens.length === 0) {
    vGens = [Permutation.identity(k)];
  }

  const vElements = dimino(vGens);
  const vOrder = vElements.length;
  const vDegree = k;

  // Classify the group name
  const factorial = (n) => n <= 1 ? 1 : n * factorial(n - 1);
  const labelSet = `{${labels.join(',')}}`;
  let vGroupName = 'trivial';
  if (vDegree >= 2 && vOrder > 1) {
    if (vOrder === factorial(vDegree)) {
      vGroupName = `S${vDegree}${labelSet}`;
    } else if (vOrder === vDegree && vDegree >= 3) {
      vGroupName = `C${vDegree}${labelSet}`;
    } else if (vOrder === 2 * vDegree && vDegree >= 3) {
      vGroupName = `D${vDegree}${labelSet}`;
    } else if (vOrder === 2) {
      const gen = vGens[0];
      const cycles = gen.cyclicForm();
      const allTwoCycles = cycles.every(c => c.length === 2);
      if (allTwoCycles && cycles.length > 1) {
        const orbitParts = cycles.map(c => `S2{${c.map(i => labels[i]).join(',')}}`);
        vGroupName = orbitParts.join('\u00d7');
      } else {
        vGroupName = `S2${labelSet}`;
      }
    } else {
      const genStr = vGens.map(g => g.cycleNotation(labels)).join(', ');
      vGroupName = `PermGroup\u27e8${genStr}\u27e9`;
    }
  }

  return {
    vLabels: labels,
    wLabels: [],
    vGenerators: vGens,
    vGeneratorsAll: vGens,
    wGenerators: [],
    vElements,
    wElements: [],
    vOrder,
    vGroupName,
    vDegree,
    declared: true,
  };
}

// ─── Burnside Counting ───────────────────────────────────────────

export function computeBurnside(group, dimensionN) {
  const fullDegree = group.fullDegree ?? group.allLabels?.length ?? 0;
  const fullElements = group.fullElements ?? (fullDegree > 0 ? [Permutation.identity(fullDegree)] : []);
  const fullSizes = Array(fullDegree).fill(dimensionN);
  const fullResult = fullDegree > 0
    ? burnsideCount(fullElements, fullSizes)
    : { perElement: [], totalFixed: 1, uniqueCount: 1 };
  const totalTupleCount = Math.pow(dimensionN, fullDegree);
  const orbitCount = fullResult.uniqueCount;

  const vDegree = group.vDegree ?? group.vLabels?.length ?? 0;
  const vElements = group.vElements ?? (vDegree > 0 ? [Permutation.identity(vDegree)] : []);
  const vSizes = Array(vDegree).fill(dimensionN);
  const vResult = vDegree > 0
    ? burnsideCount(vElements, vSizes)
    : { perElement: [], totalFixed: 1, uniqueCount: 1 };
  const totalCount = Math.pow(dimensionN, vDegree);
  const ratio = totalCount > 0 ? vResult.uniqueCount / totalCount : 1;

  const wDegree = group.wDegree ?? group.wLabels?.length ?? 0;
  const wOrder = group.wOrder ?? group.wElements?.length ?? 1;
  const wElements = group.wElements ?? (wDegree > 0 ? [Permutation.identity(wDegree)] : []);
  const wSizes = Array(wDegree).fill(dimensionN);
  const wResult = (wDegree > 0 && wOrder > 1)
    ? burnsideCount(wElements, wSizes)
    : {
        perElement: [],
        totalFixed: wDegree > 0 ? Math.pow(dimensionN, wDegree) : 1,
        uniqueCount: wDegree > 0 ? Math.pow(dimensionN, wDegree) : 1,
      };
  const wTotalCount = Math.pow(dimensionN, wDegree);
  const wRatio = wTotalCount > 0 ? wResult.uniqueCount / wTotalCount : 1;

  return {
    // Legacy split V/W fields for existing callers.
    perElement: vResult.perElement,
    totalFixed: vResult.totalFixed,
    uniqueCount: vResult.uniqueCount,
    totalCount,
    ratio,
    wPerElement: wResult.perElement,
    wUniqueCount: wResult.uniqueCount,
    wTotalCount,
    wRatio,
    wHasSymmetry: wOrder > 1,
    // New full-group fields for the Task 3 model.
    fullPerElement: fullResult.perElement,
    fullTotalFixed: fullResult.totalFixed,
    orbitCount,
    totalTupleCount,
    dimensionN,
  };
}

// ─── Cost Reduction ──────────────────────────────────────────────

export function computeCostReduction(burnside, group, numTerms = 2) {
  const { uniqueCount, totalCount, ratio } = burnside;

  // Total contraction size = V * W (output elements × summed elements)
  const allCount = totalCount * burnside.wTotalCount;
  // FMA=2 textbook convention: each multiply-add fused into 2 ops
  // (1 multiply + 1 add counted separately). For a binary contraction
  // the dense baseline is 2 * M*N*K (off-by-one not applied at this
  // per-step approximation; matches Python helpers.flop_count under FMA=2).
  const opFactor = Math.max(1, numTerms - 1) + 1;
  const denseCost = Math.max(1, opFactor * allCount);

  // V-only reduction: reduce output side by V symmetry
  const vReducedCost = Math.max(1, Math.round(denseCost * ratio));

  // V+W combined reduction: also reduce inner loop by W symmetry
  const combinedRatio = ratio * burnside.wRatio;
  const combinedReducedCost = Math.max(1, Math.round(denseCost * combinedRatio));

  return {
    denseCost,
    // V-side only
    reducedCost: vReducedCost,
    ratio,
    speedup: denseCost / vReducedCost,
    uniqueCount,
    totalCount,
    groupName: group.vGroupName,
    groupOrder: group.vOrder,
    // W-side savings (additional, shown separately)
    wReducedCost: combinedReducedCost,
    wRatio: burnside.wRatio,
    wSpeedup: denseCost / combinedReducedCost,
    wUniqueCount: burnside.wUniqueCount,
    wTotalCount: burnside.wTotalCount,
    wHasSymmetry: burnside.wHasSymmetry,
    wGroupName: group.wGroupName || 'trivial',
    // Combined
    combinedRatio,
    combinedReducedCost,
  };
}

export { analyzeExample } from './pipeline.js';
