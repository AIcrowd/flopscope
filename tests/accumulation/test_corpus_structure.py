from tests.accumulation._corpus import CORPUS


def test_corpus_has_at_least_one_case_per_regime():
    all_regimes = set()
    for case in CORPUS:
        all_regimes.update(case.expected_regimes)
    expected = {
        "trivial",
        "functionalProjection",
        "singleton",
        "young",
        "partitionCount",
    }
    assert expected <= all_regimes, f"Missing regime coverage: {expected - all_regimes}"


def test_corpus_case_ids_unique():
    ids = [c.case_id for c in CORPUS]
    assert len(ids) == len(set(ids)), "duplicate case_id"


def test_corpus_sizes_align_with_subscripts():
    for case in CORPUS:
        if not case.subscripts:
            continue
        labels_in_subs = set()
        for part in case.subscripts.split(","):
            labels_in_subs.update(part)
        labels_in_subs.update(case.output)
        for lbl in labels_in_subs:
            assert lbl in case.sizes_by_label, f"{case.case_id}: missing size for {lbl}"
