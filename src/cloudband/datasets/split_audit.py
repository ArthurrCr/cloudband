"""Check that the CloudSEN12+ splits keep every location and both processing levels apart.

Training and validation use L1C and L2A, which are the same patches before and after
atmospheric correction. That is sound only if both versions of a patch fall in the same
split and no location (ROI) appears in two splits; otherwise a test scene would have
been seen in training at the other level, or at another date of the same place.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from cloudband.datasets.cloudsen12 import ID_COLUMN, ROI_COLUMN, require_columns


@dataclass(frozen=True)
class SplitAudit:
    leaked_rois: dict[str, tuple[str, ...]]
    unpaired: dict[str, tuple[int, int]]
    ids_comparable: bool
    rois_checked: bool
    rois_per_split: dict[str, int]
    patches_per_split: dict[str, dict[str, int]]

    @property
    def problems(self) -> list[str]:
        found = []
        if self.leaked_rois:
            sample = ", ".join(
                f"{roi} in {'/'.join(splits)}"
                for roi, splits in list(self.leaked_rois.items())[:5]
            )
            found.append(f"{len(self.leaked_rois)} ROI(s) appear in more than one split: {sample}")
        for split, (only_first, only_second) in self.unpaired.items():
            found.append(
                f"{split}: {only_first} patch(es) only in L1C and {only_second} only in L2A"
            )
        return found

    @property
    def summary(self) -> str:
        lines = []
        for split, counts in self.patches_per_split.items():
            levels = ", ".join(f"{level} {count}" for level, count in counts.items())
            rois = f"{self.rois_per_split[split]} ROIs, " if self.rois_checked else ""
            lines.append(f"{split}: {rois}patches {levels}")
        if not self.rois_checked:
            lines.append("the tables have no roi_id column, so the ROI check did not run")
        if not self.ids_comparable:
            lines.append(
                "the L1C and L2A tables share no patch id, so the two levels could not be paired"
            )
        if not self.problems:
            checks = (["no ROI is shared between splits"] if self.rois_checked else []) + (
                ["every patch has both levels"] if self.ids_comparable else []
            )
            if checks:
                lines.append(" and ".join(checks))
        lines.extend(self.problems)
        return "\n".join(lines)


def audit_splits(tables: Mapping[tuple[str, str], pd.DataFrame]) -> SplitAudit:
    """Audit tables keyed by (processing level, split), such as ("L1C", "train").

    A ROI counts as leaked when it appears in more than one split, at any level. Within a
    split, the patch ids of the two levels must be the same set. When the levels share
    no id at all the naming differs between them, so that comparison is reported as not
    possible instead of as a failure; the same goes for the ROI check when a table has no
    ROI column.
    """
    for table in tables.values():
        require_columns(table, [ID_COLUMN])
    rois_checked = all(ROI_COLUMN in table.columns for table in tables.values())

    splits_of_roi: dict[str, set[str]] = {}
    rois_per_split: dict[str, set[str]] = {}
    ids: dict[str, dict[str, set[str]]] = {}
    patches: dict[str, dict[str, int]] = {}
    for (level, split), table in tables.items():
        rois = table[ROI_COLUMN].astype(str) if rois_checked else ()
        for roi in rois:
            splits_of_roi.setdefault(roi, set()).add(split)
        rois_per_split.setdefault(split, set()).update(rois)
        ids.setdefault(split, {})[level] = set(table[ID_COLUMN].astype(str))
        patches.setdefault(split, {})[level] = len(table)

    leaked = {
        roi: tuple(sorted(splits))
        for roi, splits in sorted(splits_of_roi.items())
        if len(splits) > 1
    }

    paired_splits = {split: by_level for split, by_level in ids.items() if len(by_level) == 2}
    shared = any(
        set.intersection(*by_level.values()) for by_level in paired_splits.values()
    )
    unpaired = {}
    if shared:
        for split, by_level in paired_splits.items():
            first, second = (by_level[level] for level in sorted(by_level))
            if first != second:
                unpaired[split] = (len(first - second), len(second - first))

    return SplitAudit(
        leaked_rois=leaked,
        unpaired=unpaired,
        ids_comparable=shared or not paired_splits,
        rois_checked=rois_checked,
        rois_per_split={split: len(rois) for split, rois in rois_per_split.items()},
        patches_per_split=patches,
    )
