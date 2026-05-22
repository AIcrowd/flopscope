# JS Oracle

Node.js bridge to the canonical JS symmetry-aware einsum engine for Python parity tests.

## Usage

`run.mjs` reads a JSON spec from stdin, calls `analyzeExample()` from the JS engine pipeline, and emits a JSON result to stdout.

### Input

```json
{
  "subscripts": "ij,jk",
  "output": "ik",
  "operand_names": ["A", "B"],
  "per_op_symmetry": [null, null],
  "sizes_by_label": {"i": 3, "j": 3, "k": 3}
}
```

### Output

```json
{
  "components": [...],
  "total": 54,
  "mu": 27,
  "alpha": 27,
  "mTotal": 27,
  "denseBaseline": 27,
  "numTerms": 2
}
```

### Smoke test

```bash
echo '{"subscripts":"ij,jk","output":"ik","operand_names":["A","B"],"per_op_symmetry":[null,null],"sizes_by_label":{"i":3,"j":3,"k":3}}' | node run.mjs
```

## Python wrapper

The `_js_oracle.py` module (one directory up) wraps this script in a subprocess. It is imported by `test_js_parity.py` and skipped automatically when Node is not on PATH.
