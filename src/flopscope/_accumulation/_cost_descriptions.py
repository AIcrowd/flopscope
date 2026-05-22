"""LaTeX and human-readable descriptions, built on demand from regime metadata."""

from __future__ import annotations

LATEX_BY_REGIME: dict[str, str] = {
    "trivial": r"\alpha = M = |X| = \prod_{\ell \in L} n_\ell",
    "functionalProjection": r"\alpha = M = |X / G|",
    "singleton": (
        r"\alpha = \frac{n_\Omega}{|G|}\sum_{g}\left(\prod_{c \in R} n_c\right)"
        r"\left(n_\Omega^{c_\Omega(g)} - (n_\Omega-1)^{c_\Omega(g)}\right)"
    ),
    "young": r"\alpha = \binom{n_L+|V|-1}{|V|}\binom{n_L+|W|-1}{|W|}",
    "partitionCount": (
        r"\alpha = \sum_{\tilde{x}\in P_{\mathrm{typed}}(L)/G} "
        r"\frac{\prod_s (n_s)_{b_s(\tilde{x})}}{|\overline{G}_{\tilde{x}}|}"
        r"\,|A_{\tilde{x}}/H|"
    ),
    "unavailable": r"\alpha = \text{unavailable}",
}

LATEX_SYMBOLIC_BY_REGIME: dict[str, str] = {
    "trivial": r"\alpha = M",
    "functionalProjection": r"\alpha = M",
    "singleton": r"\alpha = \#\{(O,Q): \pi_V(O)\cap Q\ne\varnothing\},\ |V|=1",
    "young": r"\alpha = |\mathrm{Multiset}_n(V)|\,|\mathrm{Multiset}_n(W)|",
    "partitionCount": r"\alpha = \#\{(O,Q): \pi_V(O)\cap Q\ne\varnothing\}",
    "unavailable": r"\alpha = \text{unavailable}",
}


def describe_component(component) -> dict[str, str]:
    """LaTeX strings for a single component."""
    return {
        "latex": LATEX_BY_REGIME.get(component.regime_id, ""),
        "latex_symbolic": LATEX_SYMBOLIC_BY_REGIME.get(component.regime_id, ""),
    }


def describe_total(cost) -> dict:
    """Summary dict for a whole-einsum AccumulationCost."""
    return {
        "total": cost.total,
        "mu": cost.mu,
        "alpha": cost.alpha,
        "m_total": cost.m_total,
        "dense_baseline": cost.dense_baseline,
        "num_terms": cost.num_terms,
        "savings_ratio": cost.savings_ratio,
        "fallback_used": cost.fallback_used,
        "unavailable_components": cost.unavailable_components,
        "per_component": [
            {
                "labels": c.labels,
                "m": c.m,
                "alpha": c.alpha,
                "regime_id": c.regime_id,
                "shape": c.shape,
                "group_name": c.group_name,
                **c.describe(),
            }
            for c in cost.per_component
        ],
    }
