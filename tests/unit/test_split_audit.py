import pandas as pd
import pytest

from cloudband.datasets.split_audit import audit_splits


def table(ids, rois):
    return pd.DataFrame({"tortilla:id": ids, "roi_id": rois})


def clean_tables():
    train = table(["a", "b", "c"], ["r1", "r1", "r2"])
    test = table(["d", "e"], ["r3", "r3"])
    return {
        ("L1C", "train"): train,
        ("L2A", "train"): train.copy(),
        ("L1C", "test"): test,
        ("L2A", "test"): test.copy(),
    }


def test_disjoint_rois_and_matching_levels_have_no_problem():
    audit = audit_splits(clean_tables())

    assert audit.problems == []
    assert audit.ids_comparable
    assert audit.rois_checked
    assert "no ROI is shared between splits and every patch has both levels" in audit.summary
    assert audit.rois_per_split == {"train": 2, "test": 1}
    assert audit.patches_per_split["train"] == {"L1C": 3, "L2A": 3}


def test_a_roi_in_two_splits_is_reported():
    tables = clean_tables()
    tables[("L1C", "test")] = table(["d", "e"], ["r3", "r1"])

    audit = audit_splits(tables)

    assert audit.leaked_rois == {"r1": ("test", "train")}
    assert any("more than one split" in problem for problem in audit.problems)


def test_a_roi_leaked_only_at_one_level_is_still_reported():
    tables = clean_tables()
    tables[("L2A", "test")] = table(["d", "e"], ["r3", "r2"])

    assert "r2" in audit_splits(tables).leaked_rois


def test_a_patch_present_at_one_level_only_is_reported():
    tables = clean_tables()
    tables[("L2A", "train")] = table(["a", "b"], ["r1", "r1"])

    audit = audit_splits(tables)

    assert audit.unpaired == {"train": (1, 0)}
    assert any("only in L1C" in problem for problem in audit.problems)


def test_levels_named_differently_are_not_taken_for_a_mismatch():
    tables = clean_tables()
    tables[("L2A", "train")] = table(["A", "B", "C"], ["r1", "r1", "r2"])
    tables[("L2A", "test")] = table(["D", "E"], ["r3", "r3"])

    audit = audit_splits(tables)

    assert not audit.ids_comparable
    assert audit.unpaired == {}
    assert audit.problems == []
    assert "could not be paired" in audit.summary


def test_without_a_roi_column_the_roi_check_is_skipped_not_failed():
    tables = {
        ("L1C", "train"): pd.DataFrame({"tortilla:id": ["a", "b"]}),
        ("L2A", "train"): pd.DataFrame({"tortilla:id": ["a", "b"]}),
    }

    audit = audit_splits(tables)

    assert not audit.rois_checked
    assert audit.leaked_rois == {}
    assert audit.problems == []
    assert "ROI check did not run" in audit.summary


def test_missing_id_column_fails_loudly():
    with pytest.raises(ValueError, match="tortilla:id"):
        audit_splits({("L1C", "train"): pd.DataFrame({"roi_id": ["r1"]})})
