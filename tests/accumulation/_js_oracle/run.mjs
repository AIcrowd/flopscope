// Reads JSON spec from stdin, calls analyzeExample(), emits JSON to stdout.
//
// Input shape:
//   {
//     "subscripts": "ij,jk",
//     "output": "ik",
//     "operand_names": ["A", "B"],
//     "per_op_symmetry": [null, null] | [{type, axes, generators?}, ...],
//     "sizes_by_label": {"i": 3, "j": 3, "k": 3}
//   }

import { analyzeExample } from '../../../website/components/symmetry-aware-einsum-contractions/engine/pipeline.js';
import { aggregateComponentCosts } from '../../../website/components/symmetry-aware-einsum-contractions/engine/costModel.js';

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString('utf8');
}

function example_from(input) {
  const variables = input.operand_names.map((name, idx) => ({
    name,
    rank: input.subscripts.split(',')[idx].length,
    symmetry: input.per_op_symmetry?.[idx] ? input.per_op_symmetry[idx].type ?? input.per_op_symmetry[idx] : 'none',
    symAxes: input.per_op_symmetry?.[idx]?.axes ?? null,
    generators: input.per_op_symmetry?.[idx]?.generators ?? '',
  }));
  return {
    id: 'parity-input',
    expression: {
      subscripts: input.subscripts,
      output: input.output,
      operandNames: input.operand_names.join(', '),
    },
    variables,
    labelSizes: input.sizes_by_label,
  };
}

const inputJson = await readStdin();
const input = JSON.parse(inputJson);
const dimensionN = Math.max(...Object.values(input.sizes_by_label));
const example = example_from(input);

const analysis = analyzeExample(example, dimensionN);
const numTerms = input.subscripts.split(',').length;
const aggregated = aggregateComponentCosts(analysis.componentData.components, numTerms);

const output = {
  components: analysis.componentData.components.map((c) => ({
    labels: c.labels,
    va: c.va,
    wa: c.wa,
    sizes: c.sizes,
    m: c.multiplication?.count ?? null,
    alpha: c.accumulation?.count ?? null,
    regimeId: c.accumulation?.regimeId ?? null,
    shape: c.shape ?? null,
    groupName: c.groupName,
    groupOrder: c.order,
    subTrace: c.accumulation?.trace ?? [],
  })),
  total: aggregated ? (numTerms - 1) * aggregated.mTotal + aggregated.alpha : null,
  mu: aggregated?.mu ?? null,
  alpha: aggregated?.alpha ?? null,
  mTotal: aggregated?.mTotal ?? null,
  denseBaseline: Object.values(input.sizes_by_label).reduce((a, b) => a * b, 1),
  numTerms,
};

process.stdout.write(JSON.stringify(output));
