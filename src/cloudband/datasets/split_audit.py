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
            lines.append(f"{split}: {self.rois_per_split[split]} ROIs, patches {levels}")
        if not self.ids_comparable:
            lines.append(
                "the L1C and L2A tables share no patch id, so the two levels could not be "
                "paired; only the ROI check ran"
            )
        lines.extend(self.problems or ["no ROI is shared between splits"
                                       + (" and every patch has both levels"
                                          if self.ids_comparable else "")])
        return "\n".join(lines)


def audit_splits(tables: Mapping[tuple[str, str], pd.DataFrame]) -> SplitAudit:
    """Audit tables keyed by (processing level, split), such as ("L1C", "train").

    A ROI counts as leaked when it appears in more than one split, at any level. Within a
    split, the patch ids of the two levels must be the same set. When the levels share
    no id at all the naming differs between them, so that comparison is reported as not
    possible instead of as a failure.
    """
    for table in tables.values():
        require_columns(table, [ID_COLUMN, ROI_COLUMN])

    splits_of_roi: dict[str, set[str]] = {}
    rois_per_split: dict[str, set[str]] = {}
    ids: dict[str, dict[str, set[str]]] = {}
    patches: dict[str, dict[str, int]] = {}
    for (level, split), table in tables.items():
        for roi in table[ROI_COLUMN].astype(str):
            splits_of_roi.setdefault(roi, set()).add(split)
        rois_per_split.setdefault(split, set()).update(table[ROI_COLUMN].astype(str))
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
        rois_per_split={split: len(rois) for split, rois in rois_per_split.items()},
        patches_per_split=patches,
    )
