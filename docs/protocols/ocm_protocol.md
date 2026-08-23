# OmniCloudMask protocol

Source: Wright, N. et al. (2025). Training sensor-agnostic deep learning models for remote
sensing: achieving state-of-the-art cloud and cloud shadow identification with OmniCloudMask.
Remote Sensing of Environment 322, 114694. Open access, CC BY.

Code: github.com/DPIRD-DMA/OmniCloudMask (MIT). Package on PyPI.

Status column: `stated` is literal in the paper, `inferred` is a reading not stated outright,
`absent` is unspecified.

## Metric and experiments

Section 2.5, Eq. 4:

    PA  = TP / (TP + FN)
    BOA = 0.5 * ( PA + TN / (TN + FP) )

Binary balanced accuracy. Identical to Eq. 6 of the STUPmask paper; both cite Skakun et al.
(2022).

Three one-vs-rest experiments: clear, cloud, shadow. When scoring one class, the other two are
merged into the negative class.

Section 2.3: thick and thin cloud are merged into a single cloud class for evaluation. The
model emits four classes, the evaluation uses three.

Eq. 3 contains a typo, writing `OA = (TP + FN)/...` where it should read `TP + TN`. The
implementation is validated against sklearn and against the published confusion matrices rather
than the printed formulas. See D10.

## Bands

Section 2.7 is literal: trained on B04 (red, 10 m), B03 (green, 10 m) and B8A (NIR, 20 m),
selected by an iterative process where other channels were removed from the training set and
only those whose removal degraded performance were kept.

B8A is the outcome of channel selection. The package documentation separately notes that B8A
is loaded rather than B08 because the 20 m file is smaller; that describes the loader, not the
model. See `docs/protocols/DIVERGENCES.md` D8.

OCM uses B8A at 864.5 nm while STUPmask most likely uses B08 at 840 nm. These are two distinct
R-G-NIR combinations among the 286, each canonised by a different paper. Track both separately
in the ranking.

Zero-based positions in the CloudSEN12+ L1C stack: B04 is 3, B03 is 2, B8A is 8. Verified
against the band table in the collection card and pinned in `pipelines/phase0.py`.

## Training data

Section 2.3: CloudSEN12, splits of 8,490 train / 535 validation / 975 test. Each patch ships at
both L1C and L2A processing levels and both were used, giving 16,980 training and 1,070
validation patches.

## Preprocessing

| Item | Value |
| --- | --- |
| Canvas | 509x509 native. No resize to 512 |
| Normalisation | Dynamic Z-score: mean and standard deviation per input and per channel |
| No-data | Excluded from the statistics, then set to zero |
| Mixed resolution | Random per-batch resampling, 9 m (565 px) to 50 m (102 px), bilinear |

The paper never mentions 512. That figure belongs to two other places: STUPmask resizes 509 to
512 by bilinear interpolation because Swin needs dimensions divisible by 32, and CloudSEN12+
stores patches on a 512 canvas padded with zeros on the left and bottom sides.

Section 2.2 justifies mixed resolution by the ability of CNN based models to process arbitrary
input sizes. Swin is not a CNN. See ADR-0023, which narrows the range to 9-22 m for both
architectures.

## Training

| Item | Value |
| --- | --- |
| Framework | Fastai `fine_tune` |
| Epochs | 50 frozen plus 50 unfrozen |
| Initial learning rate | 0.005 |
| Batch | 4 with gradient accumulation 32, effective 128 |
| Loss | Cross entropy |
| Checkpoint | Lowest validation cross entropy loss |
| Architecture | Two Fastai Dynamic U-Nets |
| Backbones | RegNetY 004 and ConvNeXtV2 nano, ImageNet weights from timm |
| Ensemble | Soft voting, mean predicted confidence per pixel and class |
| Runtime | 25 h on a Ryzen 5950X and an RTX 4090 |

Augmentations, section 2.9:

| Augmentation | Probability | Detail |
| --- | --- | --- |
| Image tear | 10 % per batch | Shifts part of the image and its label by 0-10 px |
| Random rectangle | 60 % per batch | Random fill value, not zero; one, two or three channels; label unchanged |
| Scene edge | 10 % per batch | Zeros along the edge, label deliberately unchanged |
| Rotations and flips | 100 % | 0/90/180/270 plus vertical and horizontal flips |

The training protocol is fully specified here, in contrast to roughly 30 % for STUPmask.

## Inference

Inference has no random component, so there is no seed. The configuration below must be
recorded with every result.

| Parameter | Package default | Effect |
| --- | --- | --- |
| `patch_size` | 1000 | Tile size a large scene is split into before inference |
| `patch_overlap` | 300 | Borders of overlapping tiles reconciled by gradient merging |
| `inference_dtype` | float32 | Changes marginal pixels |
| `model_version` | latest available | Distinct weight sets |
| Ensemble | Two models | Disabling changes the result |

Tiling is a separate mechanism from the mixed resolution resampling used during training. The
defaults suit a full Sentinel-2 scene of 10,980 px and have no effect on a 509 px patch, which
fits in a single tile: the package reduces them to 509 and 254 and emits a warning. Passing
`patch_size=509, patch_overlap=0` for CloudSEN12+ patches produces the same result and lets the
manifest record what actually ran.

Model versions 1.0 to 3.0 require fastai to be installed. Version 1.0 produced the values
reported in the paper.

## Fidelity targets

Two distinct references, both verifiable, both measured on PixBox Sentinel-2.

Paper, Table A.1:

| Class | TP | TN | FP | FN | BOA |
| --- | --- | --- | --- | --- | --- |
| clear | 7,791 | 8,186 | 868 | 506 | 92.2 % |
| cloud | 7,170 | 8,683 | 499 | 999 | 91.2 % |
| shadow | 778 | 15,860 | 245 | 468 | 80.5 % |

Reference notebook in the repository, OCM 1.7.0, same dataset:

| Class | TP | TN | FP | FN | BOA |
| --- | --- | --- | --- | --- | --- |
| clear | 7,837 | 8,184 | 870 | 460 | 92.42 % |
| cloud | 7,185 | 8,731 | 451 | 984 | 91.52 % |
| shadow | 798 | 15,895 | 210 | 448 | 81.37 % |

Drift of +0.22, +0.32 and +0.87 percentage points. The paper values require model version 1.0,
which needs fastai installed. Pin `omnicloudmask==1.7.0` and state which target applies to a
given run.

Paper, PixBox Landsat 8, Table A.2: clear 91.5 %, cloud 91.5 %, shadow 75.2 %, over 18,830
pixels. That total confirms the Zenodo record against the 20,500 cited by CMIX.

All fifteen matrices are parametrised cases in `tests/contract/test_metrics.py`.

### Verified reproduction

Package 1.7.0, latest weights, run over the 28 available scenes. The absent scene leaves 16,801
scorable pixels. Reference positives per class come out at 8,148 clear, 7,823 cloud and 1,164
shadow, matching the published counts minus the dropped pixels exactly.

| Class | TP | TN | FP | FN | BOA | Notebook 1.7.0 | Difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clear | 7,690 | 7,815 | 838 | 458 | 92.35 % | 92.42 % | -0.07 |
| cloud | 6,899 | 8,530 | 448 | 924 | 91.60 % | 91.52 % | +0.08 |
| shadow | 759 | 15,470 | 167 | 405 | 82.07 % | 81.37 % | +0.70 |

All three fall inside the tolerance band the missing scene implies: clear 91.53 to 93.29, cloud
89.40 to 93.47, shadow 78.08 to 83.63. The band was computed before the run, from the published
matrices with the dropped pixels counted first as all correct and then as all incorrect.

Shadow carries the largest deviation, which follows from the loss falling unevenly: the missing
scene holds 6.58 % of all shadow pixels against 4.24 % of cloud.

Shadow sensitivity is 65.21 % against specificity above 99 %. The same asymmetry appears on the
CloudSEN12+ run and is described in the paper, so the failure mode reproduces across three
independent collections.

The paper reports results on three test sets only: PixBox Sentinel-2, PixBox Landsat 8 and a
PlanetScope collection of 5,223 hand-labelled pixels at 96.9 / 98.8 / 97.4. All three sit
outside the training domain, which is what the sensor-agnostic claim rests on.

No result is published on the CloudSEN12 test split, because CloudSEN12 is the training set. A
CloudSEN12+ run therefore has no published target and cannot be a fidelity check. It measures
in-domain performance, which nobody has reported.

Fidelity against the published values requires PixBox and nothing else. It needs the 29
Sentinel-2 L1C products the pixel indices address.

## In-domain baseline on CloudSEN12+

Measured with package 1.7.0, latest weights, over the 975 test patches of the high quality p509
subset, 252,603,975 pixels. The three reference classes partition the pixels exactly.

| Class | TP | TN | FP | FN | Sensitivity | Specificity | BOA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clear | 127,851,868 | 110,577,526 | 8,067,248 | 6,107,333 | 95.44 | 93.20 | 94.32 |
| cloud | 87,421,283 | 150,704,234 | 7,383,521 | 7,094,937 | 92.49 | 95.33 | 93.91 |
| shadow | 18,682,841 | 225,278,207 | 3,197,214 | 5,445,713 | 77.43 | 98.60 | 88.02 |

Reference composition: 53.03 % clear, 37.42 % cloud, 9.55 % shadow.

Shadow fails by omission, not by false alarm: specificity 98.60 against sensitivity 77.43. The
paper reports the same asymmetry on its own test sets, describing producer accuracy for shadow
as a relative weak point driven by false negatives. The behaviour reproduces on a different
collection.

Against the PixBox values the differences are +1.90, +2.39 and +6.65 percentage points. These
are not improvements: CloudSEN12 is inside the training domain and PixBox is not, and the
populations differ in geography and class balance.

## PixBox labels

Two independent integer attributes. A pixel can be both cloud and cloud shadow.

Sentinel-2, columns `CLOUD_CHARACTERISTICS_ID` and `SHADOW_ID`:

- cloud ids 2, 3, 4, 5, 6, 8, 9, 10, 11, 12; not cloud ids 0, 1, 7
- shadow id 3; not shadow ids 0, 1, 2, 4. Topographic shadow and shadow above cloud count as
  negatives: they are shadow, but not cloud shadow
- clear is the conjunction of both negatives, not the complement of the other two classes

Totals: 17,351 pixels, 8,297 clear, 8,169 cloud, 1,246 shadow, and 361 pixels labelled both
cloud and shadow. The three counts sum to more than the total, which is expected.

`PIXEL_X` and `PIXEL_Y` index the scene raster directly, so no coregistration step is needed.
The reference notebook carries the mapping from the 29 `PRODUCT_ID` values to L1C product names,
which the CSV does not; that mapping is the acquisition manifest.

Landsat 8 class ids are unverified. See `docs/OPEN_ISSUES.md`.

## CloudSEN12+ labels

Dense and exclusive: every pixel carries one of clear 0, thick cloud 1, thin cloud 2, cloud
shadow 3. Unlike PixBox, cloud and shadow never overlap. The reserved value 99 marks no data and
appears in scribble and nolabel patches.

The class codes were confirmed against OCM predictions on two patches, one dominated by thin
cloud and one by shadow. Reference and prediction agree within 1.9 % and 2.4 % of pixels
respectively and identify the same dominant class, which no misaligned coding would produce.

The high quality p509 subset holds exactly 8,490 train, 975 test and 535 validation patches over
2,000 ROIs, five patches each. Every ROI sits entirely within one split, so the published split
carries no spatial leakage between train and test. Grouped cross-validation is still required
within a split.

Balanced accuracy is undefined for a class absent from a scene, since specificity has no
denominator. Across the 975 test patches the annotator percentages imply 705 scenes with a
defined clear value, 767 for cloud and 655 for shadow. A paired test runs on those counts, not
on 975.

## Statistical caveat from the authors

Section 4 states that the test datasets were not generated through a probability sampling
design, which prevents computing the variance of the performance estimates and limits
statistical comparison between models.

Consequence: PixBox numbers are descriptive only, with no Wilcoxon test. Inferential comparison
is restricted to the CloudSEN12+ test split, where the scene is a well defined unit, subject to
the pairable counts above.

The authors note that a global, multi-sensor test set built on a probability sampling design
following Stehman and Foody (2019) remains a gap in the literature.

## Ablations useful as diagnostics

| Ablation | clear / cloud / shadow |
| --- | --- |
| Full | 92.2 / 91.2 / 80.5 |
| No augmentation | 90.2 / 88.7 / 80.0 |
| No mixed resolution, 10 m only | 91.3 / 90.0 / 79.9 |
| ConvNeXtV2 nano alone | 91.8 / 90.9 / 79.5 |
| RegNetY 004 alone | 91.0 / 90.2 / 80.0 |
| Min-max instead of dynamic Z-score, Landsat | 81.3 / 84.2 / 55.6 |

A reproduction landing near 90.2 on clear points at the augmentations. Dynamic Z-score matters
little on Sentinel-2 and a great deal on Landsat.