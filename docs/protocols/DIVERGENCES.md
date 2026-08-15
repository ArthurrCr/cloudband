# Divergences

Places where sources disagree, or where a project note disagrees with its cited source.
Each entry records the claims, what verification establishes, and what remains open.

Status values: `open`, `resolved`.

## Summary

| Id | Subject | Status |
| --- | --- | --- |
| D1 | STUPmask Sentinel-2 training set | open |
| D2 | CloudSEN12 test split size, 975 versus 963 | open |
| D3 | PixBox Landsat 8 pixel count, 20,500 versus 18,830 | resolved |
| D4 | STUPmask Sentinel-2 band list attributed as literal | open |
| D5 | CloudSEN12 versus CloudSEN12+ | resolved |
| D6 | TP and TN swapped in the STUPmask metric text | resolved |
| D7 | CESBIO spelled CEOBIO | resolved |
| D8 | OmniCloudMask B8A selection rationale | resolved |
| D9 | OmniCloudMask published values versus current weights | resolved |
| D10 | Overall accuracy typo in the OmniCloudMask equations | resolved |

## D1. STUPmask Sentinel-2 training set

Status: open.

Section 2.2 states that the CloudSEN12 training split of 8,942 patches is used for training,
and that the CloudSEN12 validation split of 975 patches and the S2 CMC set of 2,052 patches
are independent validation. Section 3.2 states that a single unified model was trained on a
combined dataset of Landsat 8 Biome and the Sentinel-2 Cloud Mask Catalogue, with no mention
of CloudSEN12.

The readings are incompatible. Under 2.2, CMC is held out. Under 3.2, CMC is training data and
CloudSEN12 is unused.

Impact: BOA 93.64 % on the CloudSEN12 test split is the most accessible fidelity target for
this project, since that set needs no large download. Under 3.2 it comes from a model that
never saw CloudSEN12; under 2.2 from a model trained on it. The two cannot be compared to the
same reproduction.

Working assumption until resolved: the section 2.2 reading, which is the more detailed of the
two and carries patch counts. Record the assumption in the reproduction manifest.

Closes when: the corresponding authors reply. zhanqing@umd.edu, jingwei@pku.edu.cn.

## D2. CloudSEN12 test split size

Status: open.

Section 2.2 of the STUPmask paper gives the CloudSEN12 validation set as 975 patches. Section
4.4.1 refers to the CloudSEN12 testing dataset as 963 global scenes. The difference is 12.

The gap is too large for a single-digit typo and too small for a different split. A plausible
explanation is that 12 scenes were dropped by an unstated criterion, such as missing bands or
full no-data coverage.

Established: 8,490 + 535 + 975 = 10,000, matching the annotated patch count of CloudSEN12. The
975 figure is the documented split size; 963 is the anomaly.

Closes when: the authors confirm the denominator. Ask alongside D1.

## D3. PixBox Landsat 8 pixel count

Status: resolved. The count is 18,830.

The STUPmask paper, section 4.4.1, reports 20,500 pixels, citing Paperin et al. (2021b). The
Zenodo record for that dataset, version 1.0, gives 18,830 from the same 11 products. The
OmniCloudMask paper, section 2.4.2, also gives 18,830.

Established by arithmetic: every row of OmniCloudMask Table A.2 sums to exactly 18,830 across
TP, TN, FP and FN, for all three classes and all three methods. Those matrices could not sum to
that figure if the collection held 20,500 pixels.

The 20,500 figure is either an error or refers to a superseded release.

## D4. STUPmask Sentinel-2 band list

Status: open. The correction belongs to the project notes, not to the paper.

Project note 2.2 states that the native Sentinel-2 configuration is seven bands, specifically
B02, B03, B04, B08, B10, B11 and B12, and attributes this to section 3.2 of the paper.

Section 3.2 does not enumerate those bands. It states that the selected spectral features range
from blue to SWIR wavelengths, with the TIRS channel excluded because Sentinel-2 lacks it. The
seven-band list is an inference by analogy with the Landsat 8 list, which the same section does
enumerate. The inference is sound; the attribution is not.

A second ambiguity sits inside it: whether NIR means B08 or B8A is unstated. Table 1 of the
same paper classifies B08 as NIR and B8A as red edge, which favours B08.

Closes when: project note 2.2 presents the list as inference by analogy, and the B08 reading is
recorded as a project decision.

## D5. CloudSEN12 versus CloudSEN12+

Status: resolved as a documented limitation.

The STUPmask paper cites CloudSEN12 with 9,880 ROIs and 49,400 image patches, and uses 8,942
Sentinel-2 training patches. This project uses CloudSEN12+, whose high p509 split has 8,490
training patches. The difference is 452 training patches.

Established: 9,880 x 5 = 49,400, so the paper's counts are internally consistent with the
original CloudSEN12. The 8,490 / 535 / 975 split belongs to CloudSEN12+.

Consequence: a faithful protocol reproduction should not match the published STUPmask numbers
exactly, because the training collection differs. Stated in `docs/LIMITATIONS.md`.

## D6. TP and TN swapped in the STUPmask metric text

Status: resolved. Typographical.

The text below equations 3 to 8 states that FP and FN signify correctly identified cloud and
non-cloud pixels. The first pair should read TP and TN.

The equations themselves are correct, and Eq. 6 matches OmniCloudMask Eq. 4 exactly.

## D7. CESBIO spelled CEOBIO

Status: resolved. Typographical.

Section 4.4.1 of the STUPmask paper writes "S2 CEOBIO dataset". The dataset is CESBIO, cited
correctly elsewhere as Baetens et al. (2019).

## D8. OmniCloudMask B8A selection rationale

Status: resolved. B8A is the outcome of channel selection.

The package documentation notes that B8A at 20 m is loaded instead of B08 at 10 m for faster
loading due to smaller file size, which reads as an implementation convenience.

Section 2.7 of the paper is literal: the models were trained on red B04, green B03 and
near-infrared B8A, and those channels were selected through an iterative process in which other
channels were removed from the training set, retaining only those whose removal degraded
performance.

The documentation describes how the loader fetches the band; the paper describes why that band
is in the model. Both are true and neither supersedes the other.

Consequence for the ranking: OmniCloudMask uses B8A at 864.5 nm while STUPmask most likely uses
B08 at 840 nm, so the two papers canonise two different R-G-NIR combinations among the 286.
Track both separately.

## D9. OmniCloudMask published values versus current weights

Status: resolved. Two distinct targets exist, tied to two weight versions.

The paper reports BOA on PixBox Sentinel-2 of 92.2 % clear, 91.2 % cloud and 80.5 % shadow. The
reference notebook in the repository, run with package version 1.7.0, reports 92.42 %, 91.52 %
and 81.37 % on the same dataset.

Established: both sets of confusion matrices recompute to their stated BOA, and both sum to
17,351 pixels. The drift of +0.22, +0.32 and +0.87 percentage points is an improvement in the
weights, not a difference in the metric.

Cause: `model_version` defaults to the newest available weight set. Version 1.0 produced the
paper values and requires fastai to be installed.

Consequence: pin `omnicloudmask==1.7.0` and state which target applies to a given run. Both
sets are parametrised cases in `tests/contract/test_metrics.py`.

## D10. Overall accuracy typo in the OmniCloudMask equations

Status: resolved. Typographical.

Equation 3 is printed as `OA = (TP + FN) / (TP + TN + FP + FN)`. The numerator should be
`TP + TN`. The STUPmask paper prints the same metric correctly.

Both papers carry a typographical error in their metrics section, so printed formulas are not a
sufficient reference for implementation.