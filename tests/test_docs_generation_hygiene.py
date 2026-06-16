# tests/test_docs_generation_hygiene.py
from __future__ import annotations
import importlib.util, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("gen_api_docs", ROOT / "scripts" / "generate_api_docs.py")
assert _spec is not None and _spec.loader is not None  # keep pyright happy (CI checks tests/)
gen = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gen
_spec.loader.exec_module(gen)


def test_generated_output_paths_lists_live_write_set():
    paths = {str(p) for p in gen.generated_output_paths()}
    # The two directories the live generator actually writes.
    assert any(p.endswith("website/.generated") for p in paths)
    assert any(p.endswith("website/public/api-data") for p in paths)
    # ops.json is the cost-model snapshot and stays TRACKED — never an "ignored output".
    assert not any(p.endswith("public/ops.json") for p in paths)
    # Hand-written pages must NOT be in the generated set (else we'd untrack them).
    assert not any(p.endswith("content/docs/api/index.mdx") for p in paths)
    assert not any(p.endswith("content/docs/api/numpy.mdx") for p in paths)
