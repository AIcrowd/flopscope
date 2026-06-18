from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location(
    "gen_cmp", ROOT / "scripts" / "generate_api_docs.py"
)
assert (
    _spec is not None and _spec.loader is not None
)  # keep pyright happy (CI checks tests/)
gen = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gen
_spec.loader.exec_module(gen)

SRC = (ROOT / "docs" / "reference" / "cost-model.md").read_text()


def test_frontmatter_and_source_content():
    out = gen.render_cost_model_page(SRC)
    assert out.startswith("---\n")
    assert 'title: "FLOP Counting Model"' in out
    assert "Cost model reference" in out  # carried the real content


def test_mdx_safe_no_bare_angle_or_brace_outside_code():
    out = gen.render_cost_model_page(SRC)
    body = out.split("---\n", 2)[-1]
    in_fence = False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or "`" in line:
            continue
        assert "<" not in line and "{" not in line, f"unescaped MDX token: {line!r}"


def test_not_stale_no_flop_multiplier():
    out = gen.render_cost_model_page(SRC)
    assert "flop_multiplier" not in out


def test_no_internal_links_to_removed_calibration_page():
    content = ROOT / "website" / "content" / "docs"
    offenders = []
    for mdx in content.rglob("*.mdx"):
        if "/api/" in str(mdx):  # generated API pages are not committed
            continue
        if "/docs/development/calibration/" in mdx.read_text():
            offenders.append(str(mdx.relative_to(ROOT)))
    assert not offenders, f"links to removed calibration page: {offenders}"


def test_calibration_page_removed_from_nav():
    import json

    meta = json.loads((ROOT / "website/content/docs/development/meta.json").read_text())
    assert "calibration" not in meta.get("pages", [])
