import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import katex from 'katex';

const __dirname = dirname(fileURLToPath(import.meta.url));

function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

test('MultiplicationCostCard exports a default React component', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/MultiplicationCostCard.jsx');
  assert.match(src, /export default function MultiplicationCostCard/);
});

test('MultiplicationCostCard uses the aligned M/μ/α event labels and caption', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/MultiplicationCostCard.jsx');
  assert.match(src, /Multiplication Events \(μ\)/);
  assert.match(src, /<InlineMathText>\{String\.raw`\$M\$ counts representative product values\. \$\\mu = \(k-1\)M\$ counts multiplication-chain events\. \$\\alpha\$ counts accumulation updates from product-orbit representatives into stored output representatives\.`\}<\/InlineMathText>/);
  assert.match(src, /Live for this example/i);
  assert.match(src, /\\mu\s*\\;=\\;\s*\(/);
  assert.match(src, /num_terms/);
  assert.match(src, /\\mathrm\{cycles\}\(g\)/);
  assert.match(src, /multiplicationCount/);
  assert.match(src, /InlineMathText/);
  assert.match(src, /className="explorer-support-prose mt-2"/);
  assert.match(src, /V_\{\\mathrm\{free\}\}/);
  assert.match(src, /W_\{\\mathrm\{summed\}\}/);
  assert.doesNotMatch(src, /Calculating Multiplication Cost \(μ\)/);
});

test('AccumulationHardCard exports a default React component with shared support prose and no extra pointer footer', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/AccumulationHardCard.jsx');
  assert.match(src, /export default function AccumulationHardCard/);
  // Title now uses α to match the Counting Convention band.
  assert.match(src, /Why Accumulation Cost \(α\) is Hard/);
  assert.match(src, /notationColor\('alpha_total'\)/);
  assert.match(src, /InlineMathText/);
  assert.match(src, /className="explorer-support-prose mt-2"/);
  assert.match(src, /className="explorer-support-prose mt-3"/);
  assert.doesNotMatch(src, /bg-amber-400/);
  assert.doesNotMatch(src, /See the Classification Tree below/);
  assert.match(src, /V_\{\\mathrm\{free\}\}/);
  assert.match(src, /W_\{\\mathrm\{summed\}\}/);
});

test('ComponentCostView uses the shared support prose tier under subsection headers', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx');
  assert.match(src, /editorial-two-col-divider-lg editorial-two-col-divider-lg-inset border-y border-gray-100 py-6 grid grid-cols-1 gap-6 lg:grid-cols-2/);
  assert.match(src, /id="two-cost-cards"[\s\S]*?className="[^"]*grid grid-cols-1 gap-6 lg:grid-cols-2[^"]*"/);
  assert.match(src, /<MultiplicationCostCard/);
  assert.match(src, /<AccumulationHardCard/);
});

test('ComponentCostView renders BranchingDemo before the cost cards + classification tree', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx');
  const branchingIdx = src.indexOf('<BranchingDemo');
  const muCardIdx = src.indexOf('<MultiplicationCostCard');
  const treeIdx = src.indexOf('<DecisionLadder');
  assert.ok(branchingIdx > 0, 'BranchingDemo should be present');
  assert.ok(muCardIdx > branchingIdx, 'BranchingDemo should come BEFORE MultiplicationCostCard (lifted to right after the §4 intro)');
  assert.ok(treeIdx > branchingIdx, 'BranchingDemo should come BEFORE DecisionLadder');
});

test('ComponentCostView mounts BranchingDemo full-width (single-column, no TypedPartitionDemo)', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx');
  // After the §4 partition split, BranchingDemo lives on its own row (not paired
  // with TypedPartitionDemo, which migrated to a future Partition Counting section).
  assert.match(src, /id="demos-1col"[\s\S]*?<BranchingDemo/);
  assert.doesNotMatch(src, /<TypedPartitionDemo/);
});

test('ComponentCostView exposes section-placement switches for the app layout', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx');
  assert.match(src, /showBranchingDemo = true/);
  assert.match(src, /showCostCards = true/);
  assert.match(src, /showDecisionLadder = true/);
  assert.match(src, /\{showBranchingDemo \? \(/);
  assert.match(src, /\{showCostCards \? \(/);
  assert.match(src, /\{showDecisionLadder \? \(/);
});

test('ComponentCostView imports the two new cards + renders the CLASSIFICATION TREE section', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx');
  assert.match(src, /import MultiplicationCostCard/);
  assert.match(src, /import AccumulationHardCard/);
  assert.match(src, /Classification Tree/);
  assert.match(src, /Product Orbits \(/);
  assert.match(src, /Accumulation Updates \(/);
  assert.match(src, /NotationSymbol id="m_component"/);
  assert.match(src, /NotationSymbol id="alpha_component"/);
  assert.match(src, /The dense baseline is the same direct-event convention without symmetry: one product chain and one output update for every full label assignment\./);
  assert.doesNotMatch(src, /Product orbits\s*</);
  assert.doesNotMatch(src, /Output updates\s*</);
  assert.doesNotMatch(src, /NotationSymbol id="m_total"/);
  assert.doesNotMatch(src, /NotationSymbol id="alpha_total"/);
});

test('Decision surfaces rename the scalar-output leaf to Direct Scalar Events', () => {
  const ladderSrc = read('components/symmetry-aware-einsum-contractions/components/DecisionLadder.jsx');
  const traceSrc = read('components/symmetry-aware-einsum-contractions/components/RegimeTrace.jsx');

  assert.match(ladderSrc, /Direct Scalar Events/);
  assert.match(traceSrc, /Direct Scalar Events/);
});

test('ComponentCostView passes activeLeafIds (all detected) to the DecisionLadder', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/ComponentCostView.jsx');
  assert.match(src, /activeLeafIds=/);
  // Should flatmap over components for union of regimeId + shape.
  assert.match(src, /c\.accumulation\?\.regimeId/);
  assert.match(src, /c\.shape/);
});

test('App places the O→Q, factorization, and shortcut heroes in their narrative sections', () => {
  const src = read('components/symmetry-aware-einsum-contractions/SymmetryAwareEinsumContractionsApp.jsx');
  const section3Idx = src.indexOf('§3 Projection');
  const section5Idx = src.indexOf('§5 Component Factorization');
  const section7Idx = src.indexOf('§7 Counting Shortcuts');
  const branchingIdx = src.indexOf('<BranchingDemo', section3Idx);
  const componentCostIdx = src.indexOf('<ComponentCostView', section5Idx);
  const decisionIdx = src.indexOf('<DecisionLadder', section7Idx);
  assert.ok(branchingIdx > section3Idx && branchingIdx < section5Idx, 'BranchingDemo must be a §3 O→Q hero');
  assert.ok(componentCostIdx > section5Idx && componentCostIdx < section7Idx, 'ComponentCostView summary must move to §5');
  assert.ok(decisionIdx > section7Idx, 'DecisionLadder must be the §7 shortcut hero');
  const componentCostBlock = src.slice(componentCostIdx, src.indexOf('/>', componentCostIdx) + 2);
  assert.match(componentCostBlock, /showBranchingDemo=\{false\}/);
  assert.match(componentCostBlock, /showCostCards=\{false\}/);
  assert.match(componentCostBlock, /showDecisionLadder=\{false\}/);
});


test('Section 5 piecewise brace renders as valid KaTeX', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx');
  const match = src.match(/const PIECEWISE_BRACE = String\.raw`([^`]+)`;/);

  assert.ok(match, 'expected PIECEWISE_BRACE constant in TotalCostView');

  const html = katex.renderToString(match[1], {
    throwOnError: false,
    trust: true,
  });

  assert.doesNotMatch(html, /katex-error/);
});

test('TotalCostView compares dense and symmetry-aware direct event counts with product-size formulas', () => {
  const src = read('components/symmetry-aware-einsum-contractions/components/TotalCostView.jsx');
  const assembleCost = read('components/symmetry-aware-einsum-contractions/content/main/assembleCost.js');
  const sectionNineCopy = `${src}\n${assembleCost}`;

  assert.match(src, /denseDirectEventCostFromComponents/);
  assert.match(src, /denseTupleCountFromComponents/);
  assert.match(src, /label:\s*'Dense Direct Events'/);
  assert.match(src, /label:\s*'Symmetry-Aware Direct Events'/);
  // The dense formula now subtracts the visible-axis product to apply the
  // same off-by-one correction the symmetry-aware aggregator uses.
  assert.match(src, /formula:\s*String\.raw`\((k-1|k-1)\)\\prod_\{\\ell\\in L\} n_\\ell \+ \\prod_\{\\ell\\in L\} n_\\ell - \\prod_\{\\ell\\in V\} n_\\ell`/);
  assert.doesNotMatch(src, /label:\s*'Dense Cost'/);
  assert.doesNotMatch(src, /label:\s*'Symmetry-Aware Cost'/);
  assert.doesNotMatch(src, /n\^\{\|L\|\}/);
  assert.match(src, /For each independent component, \$M_a\$ counts representative products and \$\\alpha_a\$ counts filled local \$O \\to Q\$ cells/);
  assert.match(src, /accumulation reach multiplies across independent incidence relations/);
  assert.match(src, /multiplication-chain events needed to combine each representative product across \$k\$ operands/);
  assert.match(src, /style=\{\{ textAlign: 'justify' \}\}/);
  assert.doesNotMatch(sectionNineCopy, /The result is a direct indexed scalar-event count/);
  assert.doesNotMatch(sectionNineCopy, /If an exact component count exceeds the analytic regimes or interactive budget/);
  assert.doesNotMatch(src, /The final number is/);
  assert.doesNotMatch(src, /Equivalently, for component/);
  assert.doesNotMatch(src, /SECTION_FIVE_INTRO_RESULT/);
});
