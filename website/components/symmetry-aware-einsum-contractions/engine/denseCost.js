export function denseTupleCountFromComponents(components = []) {
  if (!Array.isArray(components)) return 0;
  if (components.length === 0) return 1;
  let total = 1;
  for (const component of components) {
    const sizes = Array.isArray(component.sizes) ? component.sizes : [];
    for (const size of sizes) total *= size;
  }
  return total;
}

/**
 * Dense output-cell count: product of visible-label sizes across components.
 * This is the dense analogue of `aggregateComponentCosts.outputOrbitProduct`
 * (with no symmetry, num_output_orbits = ∏ visible_sizes per component).
 */
function denseOutputTupleCountFromComponents(components = []) {
  if (!Array.isArray(components) || components.length === 0) return 1;
  let total = 1;
  for (const component of components) {
    const labels = Array.isArray(component.labels) ? component.labels : [];
    const sizes = Array.isArray(component.sizes) ? component.sizes : [];
    const va = Array.isArray(component.va) ? component.va : [];
    if (va.length === 0) continue; // scalar output → 1 cell (no-op multiplier)
    for (const visibleLabel of va) {
      const idx = labels.indexOf(visibleLabel);
      if (idx >= 0 && sizes[idx] != null) total *= sizes[idx];
    }
  }
  return total;
}

export function denseDirectEventCostFromComponents(components = [], numTerms = 1) {
  // Dense direct-event cost with the same off-by-one correction
  // aggregateComponentCosts applies: total = (k-1)·∏n + ∏n − ∏n_visible.
  // The −∏n_visible term accounts for the free first cell of each dense
  // output orbit, keeping the symmetry-vs-dense speedup ratio honest.
  const denseTuples = denseTupleCountFromComponents(components);
  const denseOutputs = denseOutputTupleCountFromComponents(components);
  const multiplicationFactor = Math.max(numTerms - 1, 0);
  return multiplicationFactor * denseTuples + denseTuples - denseOutputs;
}

export function denseGridScalingLatex({ labelCount = 0, hasHeterogeneousSizes = false } = {}) {
  if (hasHeterogeneousSizes) return String.raw`\prod_{\ell \in L} n_\ell`;
  if (labelCount <= 0) return '1';
  if (labelCount === 1) return 'n';
  return `n^{${labelCount}}`;
}

export function hasHeterogeneousLabelSizesFromOverrides(labelSizes = {}, defaultSize) {
  const values = Object.values(labelSizes ?? {});
  if (values.length === 0) return false;
  return new Set(values).size > 1;
}
