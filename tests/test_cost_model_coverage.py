"""Cost-model completeness guards.

ROW-LEVEL: every billed registry op is reachable in ops.json (the generated
exhaustive reference), resolving ufunc/function aliases to their canonical op.
CLASS-LEVEL: every ops.json `area` is documented by a family section in
cost-model.md, so no whole op-class is silently undocumented.
DTYPE: the 'Dtype and precision' section exists, states the four-factor billing
formula verbatim, and its rate table covers every supported dtype.
"""

import importlib.util
import json
import sys
from pathlib import Path

from flopscope._registry import REGISTRY

ROOT = Path(__file__).resolve().parents[1]
OPS_JSON = ROOT / "website" / "public" / "ops.json"
COST_MODEL_MD = ROOT / "docs" / "reference" / "cost-model.md"
DEFAULT_WEIGHTS = ROOT / "src" / "flopscope" / "data" / "default_weights.json"


def _load_alias_map() -> dict[str, str]:
    # Reuse the generator's alias resolver (notes + weights.csv).
    spec = importlib.util.spec_from_file_location(
        "_gen_api_docs", ROOT / "scripts" / "generate_api_docs.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the module's @dataclass field annotations resolve
    # (Py3.14 dataclasses looks the module up via sys.modules[cls.__module__]).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.load_alias_map(REGISTRY)


def _reaches_ops_json(op: str, ops_names: set[str], alias_map: dict[str, str]) -> bool:
    """True if ``op`` — or any op in its alias chain — appears in ops.json.

    The alias map is a cost-sharing relation that can *chain* (e.g.
    ``around -> round -> rint``: ``around`` and ``round`` are not separate
    ops.json entries, but ``rint`` — the canonical they share cost with — is).
    Resolve transitively to a fixpoint; the ``seen`` set bounds any cycle so a
    cyclic chain that never reaches ops.json is still correctly flagged missing.
    """
    seen: set[str] = set()
    cur: str | None = op
    while cur is not None and cur not in seen:
        seen.add(cur)
        leaf = cur.split(".")[-1]
        if cur in ops_names or leaf in ops_names:
            return True
        cur = alias_map.get(cur) or alias_map.get(leaf)
    return False


def test_every_billed_op_is_in_ops_json_or_an_alias():
    ops = json.loads(OPS_JSON.read_text())["operations"]
    ops_names = {str(o["name"]) for o in ops}
    alias_map = _load_alias_map()
    missing = [
        op
        for op, entry in REGISTRY.items()
        if entry["category"] != "blacklisted"
        and not _reaches_ops_json(op, ops_names, alias_map)
    ]
    assert not missing, (
        f"{len(missing)} billed registry ops are neither in ops.json nor a known "
        f"alias (transitively) of one: {sorted(missing)}"
    )


AREA_MARKERS = {
    "core": "elementwise",
    "fft": "fft",
    "linalg": "linalg",
    "random": "random",
    "stats": "stats",
}


def _areas_missing_a_family_section(doc: str, areas: set[str]) -> list[str]:
    """Areas with no `### <Family>` heading under "Cost by family".

    Scoped to headings on purpose. A substring search over the whole document
    is vacuous -- every area marker also occurs in tables, op names and prose,
    so a deleted section still leaves dozens of matches behind.
    """
    body = doc[doc.index("## Cost by family") :]
    headings = [
        line[4:].strip().lower() for line in body.splitlines() if line.startswith("### ")
    ]
    return sorted(
        a for a in areas
        if not any(AREA_MARKERS.get(a, a) in h for h in headings)
    )


def test_every_ops_json_area_has_a_doc_family():
    ops = json.loads(OPS_JSON.read_text())["operations"]
    areas = {str(o["area"]) for o in ops}  # {'core','fft','linalg','random','stats'}
    missing = _areas_missing_a_family_section(COST_MODEL_MD.read_text(), areas)
    assert not missing, (
        f"ops.json areas with no cost-model.md family section: {missing}"
    )


def test_family_coverage_guard_is_not_vacuous():
    """The guard must FAIL when a family section is removed.

    It previously passed a whole-document substring search, so deleting the
    entire FFT section (2,755 chars) left it green -- "fft" still occurred 30
    times elsewhere. A guard that cannot fail is not a guard.
    """
    doc = COST_MODEL_MD.read_text()
    start = doc.index("### FFT")
    end = doc.index("### Polynomial")
    mutated = doc[:start] + doc[end:]

    areas = {str(o["area"]) for o in json.loads(OPS_JSON.read_text())["operations"]}
    missing = _areas_missing_a_family_section(mutated, areas)
    assert "fft" in missing, (
        "removing the FFT family section did not trip the coverage guard"
    )


# ---------------------------------------------------------------------------
# DTYPE-LEVEL: the 'Dtype and precision' section documents the width/complex
# pricing that the four-factor billing formula rests on.
# ---------------------------------------------------------------------------

FOUR_FACTOR_FORMULA = "charged = int(flop_cost × dtype_rate × complex_factor × weight)"


def test_cost_model_has_dtype_and_precision_section():
    doc = COST_MODEL_MD.read_text()
    assert "## Dtype and precision" in doc, (
        "cost-model.md is missing the '## Dtype and precision' section heading"
    )


def test_cost_model_states_the_four_factor_formula():
    doc = COST_MODEL_MD.read_text()
    assert FOUR_FACTOR_FORMULA in doc, (
        "cost-model.md must state the four-factor billing formula verbatim: "
        f"{FOUR_FACTOR_FORMULA!r}"
    )


def test_cost_model_dtype_table_covers_every_supported_dtype():
    # Source the dtype names from the same policy file billing reads, so a new
    # supported dtype must be added to the doc's rate table too.
    doc = COST_MODEL_MD.read_text()
    rates = json.loads(DEFAULT_WEIGHTS.read_text())["dtype_rates"]
    assert len(rates) == 18, (
        f"expected 18 supported dtypes in default_weights.json, got {len(rates)}"
    )
    missing = [name for name in rates if f"`{name}`" not in doc]
    assert not missing, (
        f"cost-model.md dtype rate table is missing a row for: {sorted(missing)}"
    )
