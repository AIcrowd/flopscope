import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

const execFileAsync = promisify(execFile);
const websiteRoot = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.dirname(websiteRoot);
const symmetryGuidePath = path.join(
  websiteRoot,
  'content/docs/guides/symmetry.mdx',
);

async function readSymmetryGuide() {
  return readFile(symmetryGuidePath, 'utf8');
}

async function runPythonInRepo(source) {
  const { stdout } = await execFileAsync(
    'uv',
    ['run', 'python', '-c', source],
    {
      cwd: repoRoot,
      maxBuffer: 1024 * 1024,
    },
  );

  return stdout;
}

test('symmetry guide documents every declaration style', async () => {
  const source = await readSymmetryGuide();

  assert.match(source, /symmetry=flops\.SymmetryGroup\.symmetric\(axes=\(0, 1\)\)/);
  assert.match(
    source,
    /symmetry=flops\.SymmetryGroup\.young\(blocks=\(\(0, 1\), \(2, 3\)\)\)/,
  );
  assert.match(
    source,
    /SymmetryGroup\.symmetric\(axes=\(0, 1, 2\)\)/,
  );
  assert.match(
    source,
    /SymmetryGroup\.cyclic\(axes=\(0, 1, 2\)\)/,
  );
  assert.match(
    source,
    /SymmetryGroup\.dihedral\(axes=\(0, 1, 2, 3\)\)/,
  );
  assert.match(
    source,
    /flops\.SymmetryGroup\.from_generators\(\s*\[\[2, 3, 0, 1\]\]/,
  );
});

test('symmetry guide includes the Reynolds section and flopscope-only indexing examples', async () => {
  const source = await readSymmetryGuide();

  assert.match(source, /Generating example data with the Reynolds operator/);
  assert.match(source, /fnp\.newaxis/);
  assert.match(source, /fnp\.array\(\[0, 1\]\)/);
});

test('symmetry guide avoids overclaiming Reynolds helper exactness', async () => {
  const source = await readSymmetryGuide();

  assert.doesNotMatch(source, /How to generate exact example tensors/);
  assert.doesNotMatch(
    source,
    /Generating exact example data with the Reynolds operator/,
  );
  assert.doesNotMatch(
    source,
    /it produces data that exactly satisfies the declared symmetry/,
  );
  assert.match(source, /validation tolerances/);
});

test('symmetry guide explains why the einsum savings example is cheaper', async () => {
  const source = await readSymmetryGuide();

  assert.doesNotMatch(source, /the output is symmetric in `\(0, 1\)`/);
  assert.match(source, /repeated `x` operand/);
  assert.match(
    source,
    /does not reduce that `einsum`'s own cost by itself/,
  );
});

test('symmetry guide documents exact unary pointwise preservation for non-full groups', async () => {
  const source = await readSymmetryGuide();

  assert.match(source, /Unary pointwise ops preserve symmetry-aware costs and keep the same exact group/);
  assert.match(source, /Unary pointwise ops preserve symmetry-aware costs/);
  assert.match(source, /including non-full groups such as `C_k` or `D_k`/);
  assert.match(source, /custom exact groups/);
  assert.doesNotMatch(source, /current implementation widens the metadata to full symmetry/);
});

test('symmetry guide representative propagation claims match runtime behavior', async () => {
  const stdout = await runPythonInRepo(`
import json
import flopscope as flops
import flopscope.numpy as fnp

flops.configure(symmetry_warnings=False)

def symmetrize_with_group(shape, symmetry_group):
    raw_data = fnp.random.randn(*shape)
    group_axes = (
        symmetry_group.axes
        if symmetry_group.axes is not None
        else tuple(range(symmetry_group.degree))
    )
    accumulated = fnp.zeros_like(raw_data)

    for group_element in symmetry_group.elements():
        axis_permutation = list(range(len(shape)))
        for group_index, tensor_axis in enumerate(group_axes):
            axis_permutation[tensor_axis] = group_axes[group_element.array_form[group_index]]
        accumulated = accumulated + fnp.transpose(raw_data, axis_permutation)

    return flops.as_symmetric(
        accumulated / symmetry_group.order(),
        symmetry=symmetry_group,
    )

s2_matrix = flops.as_symmetric(
    fnp.ones((6, 6)),
    symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
)
row_slice = s2_matrix[0]
advanced_index_slice = s2_matrix[fnp.array([0, 1])]

s3_group = flops.SymmetryGroup.symmetric(axes=(0, 1, 2))
c3_group = flops.SymmetryGroup.cyclic(axes=(0, 1, 2))
c4_group = flops.SymmetryGroup.cyclic(axes=(0, 1, 2, 3))

s3_tensor = symmetrize_with_group((4, 4, 4), s3_group)
c3_tensor = symmetrize_with_group((4, 4, 4), c3_group)
c4_tensor = symmetrize_with_group((4, 4, 4, 4), c4_group)
d4_group = flops.SymmetryGroup.dihedral(axes=(0, 1, 2, 3))
d4_tensor = symmetrize_with_group((4, 4, 4, 4), d4_group)

s3_slice = s3_tensor[:, :, 0]
c3_slice = c3_tensor[:, :, 0]
c4_reduced = fnp.sum(c4_tensor, axis=(1, 3))
intersection_tensor = fnp.add(s3_tensor, c3_tensor)
unary_c3 = fnp.exp(c3_tensor)
unary_d4 = fnp.exp(d4_tensor)

with flops.BudgetContext(flop_budget=10**8) as budget:
    repeated_input = fnp.ones((5, 3))
    repeated_einsum = fnp.einsum(
        'ki,kj->ij',
        repeated_input,
        repeated_input,
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    repeated_einsum_cost = budget.flops_used

with flops.BudgetContext(flop_budget=10**8) as budget:
    distinct_left = fnp.ones((5, 3))
    distinct_right = fnp.ones((5, 3))
    distinct_einsum = fnp.einsum(
        'ki,kj->ij',
        distinct_left,
        distinct_right,
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    distinct_einsum_cost = budget.flops_used

left_tensor = flops.as_symmetric(
    fnp.ones((3, 3, 5, 5)),
    symmetry=flops.SymmetryGroup.young(blocks=((0, 1), (2, 3))),
)
right_tensor = flops.as_symmetric(
    fnp.ones((3, 3, 5, 5)),
    symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
)
shared_group_tensor = fnp.add(left_tensor, right_tensor)

stretched_s2_tensor = flops.as_symmetric(
    fnp.ones((1, 1, 4)),
    symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
)
plain_tensor = fnp.ones((3, 3, 4))
broadcast_sum = fnp.add(stretched_s2_tensor, plain_tensor)

print(json.dumps({
    "row_slice_type": type(row_slice).__name__,
    "advanced_index_slice_type": type(advanced_index_slice).__name__,
    "s3_slice_type": type(s3_slice).__name__,
    "s3_slice_order": s3_slice.symmetry.order(),
    "s3_slice_axes": list(s3_slice.symmetry.axes),
    "c3_slice_type": type(c3_slice).__name__,
    "c3_slice_has_symmetry": getattr(c3_slice, "symmetry", None) is not None,
    "c4_reduced_type": type(c4_reduced).__name__,
    "c4_reduced_order": c4_reduced.symmetry.order(),
    "c4_reduced_axes": list(c4_reduced.symmetry.axes),
    "intersection_type": type(intersection_tensor).__name__,
    "intersection_order": intersection_tensor.symmetry.order(),
    "shared_group_type": type(shared_group_tensor).__name__,
    "shared_group_axes": list(shared_group_tensor.symmetry.axes),
    "broadcast_has_symmetry": getattr(broadcast_sum, "symmetry", None) is not None,
    "unary_c3_order": unary_c3.symmetry.order(),
    "unary_d4_order": unary_d4.symmetry.order(),
    "repeated_einsum_cost": repeated_einsum_cost,
    "distinct_einsum_cost": distinct_einsum_cost,
    "repeated_einsum_type": type(repeated_einsum).__name__,
    "distinct_einsum_type": type(distinct_einsum).__name__,
}))
`);
  const runtimeBehavior = JSON.parse(stdout.trim());

  assert.deepEqual(runtimeBehavior, {
    row_slice_type: 'FlopscopeArray',
    advanced_index_slice_type: 'FlopscopeArray',
    s3_slice_type: 'SymmetricTensor',
    s3_slice_order: 2,
    s3_slice_axes: [0, 1],
    c3_slice_type: 'FlopscopeArray',
    c3_slice_has_symmetry: false,
    c4_reduced_type: 'SymmetricTensor',
    c4_reduced_order: 2,
    c4_reduced_axes: [0, 1],
    intersection_type: 'SymmetricTensor',
    intersection_order: 3,
    shared_group_type: 'SymmetricTensor',
    shared_group_axes: [0, 1, 2, 3],
    broadcast_has_symmetry: true,
    unary_c3_order: 3,
    unary_d4_order: 8,
    // After the off-by-one fix in einsum_accumulation_cost: the formula
    // is now `(k-1)·∏M + ∏α − ∏num_output_orbits` (mirrors the same
    // "first cell of each output orbit is a free copy" correction the
    // reduction surface already applies). For ki,kj->ij with k=5,i=j=3
    // and S_2 on the (i,j) output: ∏num_output_orbits = 6 for the
    // repeated-input case and 9 for the distinct-operand case.
    repeated_einsum_cost: 54,
    distinct_einsum_cost: 81,
    repeated_einsum_type: 'SymmetricTensor',
    distinct_einsum_type: 'SymmetricTensor',
  });
});

test('symmetry guide includes all major propagation sections', async () => {
  const source = await readSymmetryGuide();

  assert.match(source, /## Symmetry propagation at a glance/);
  assert.match(source, /## Slicing rules/);
  assert.match(source, /## Reduction rules/);
  assert.match(source, /## Binary pointwise ops and broadcasting/);
  assert.match(source, /## Warnings, conservative behavior, and re-tagging/);
  assert.match(source, /## Under the hood/);
});
