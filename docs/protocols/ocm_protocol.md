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
| `patch_size` | 1000 | Context available per patch |
| `patch_overlap` | 300 | Borders reconciled by gradient merging |
| `inference_dtype` | float32 | Changes marginal pixels |
| `model_version` | latest available | Distinct weight sets |
| Ensemble | Two models | Disabling changes the result |

Model versions 1.0 to 3.0 require fastai to be installed. Version 1.0 produced the values
reported in the paper.

## Fidelity targets

Two distinct references, both verifiable.

Paper, PixBox Sentinel-2, Table A.1:

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

## Statistical caveat from the authors

Section 4 states that the test datasets were not generated through a probability sampling
design, which prevents computing the variance of the performance estimates and limits
statistical comparison between models.

Consequence: PixBox numbers are descriptive only, with no Wilcoxon test. Inferential comparison
is restricted to the CloudSEN12+ test split, where the scene is a well defined unit and there
are 975 pairs. Project note 2.10 is scoped accordingly.

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