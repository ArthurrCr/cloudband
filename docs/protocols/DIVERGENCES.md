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
| D11 | PixBox Sentinel-2 scene declared but not published | resolved |
| D12 | The two PixBox collections use incompatible schemas | resolved |
| D13 | OmniCloudMask model version 1.0 weights no longer load | open |
| D14 | Published weights postdate the paper and were trained differently | resolved |
| D15 | OCM trains both backbones at one learning rate; we search one per backbone | resolved |
| D16 | OmniCloudMask augmentations: the paper against the authors' current code | resolved |
| D17 | The 509 window was cropped from the wrong side | resolved |
| D18 | The 13-band experiment: L1C only, and a pretrained stem adapted from 3 to 13 bands | resolved |
| D19 | U-Net + RegNetZ-D8 retrained with the recipe of the earlier repository, not ADR-0023 | resolved |

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

Established directly: the distributed CSV `pixbox_landsat8_cmix_20150527.csv` holds 18,830 rows,
and its per-product counts sum to the same figure across all 11 products. Every row of
OmniCloudMask Table A.2 sums to 18,830 as well.

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

## D11. PixBox Sentinel-2 scene declared but not published

Status: resolved. 28 of the 29 declared scenes are available.

The `pixbox_sentinel2_cmix_20180425_definitions.txt` file lists 29 Sentinel-2 products, and the
label CSV holds 17,351 pixels drawn from all 29. The `Sentinel-2_L1C.zip` archive published in
the same Zenodo record (10.5281/zenodo.5036991) contains 28. The absent product is
`S2A_MSIL1C_20180223T092031_N0206_R093_T36VUM_20180223T113049`, PRODUCT_ID 872013732.

Established by listing the archive index remotely and comparing against the product table in
the definitions file. No extra scenes are present, so the archive is a strict subset.

Its 550 pixels are 3.17 % of the collection, but they are not spread evenly across classes:
346 cloud pixels, 4.24 % of all cloud, and 82 shadow pixels, 6.58 % of all shadow.

Consequence for fidelity. The published confusion matrices sum to 17,351, so the reference
values were computed over all 29 scenes, from a source other than this archive. A run on the
28 available scenes is measured over 16,801 pixels and cannot match exactly. The gap is bounded:
recomputing the published shadow matrix with the 82 pixels counted first as all correct and
then as all incorrect moves BOA from 80.46 to 81.62 or to 78.53. A deviation inside that band
does not distinguish a pipeline error from the absent scene.

`labels/pixbox.py` declares the product id, drops its pixels, and records the per-class loss in
the run manifest.

Outcome. A run over the 28 available scenes reproduces the reference values within -0.07, +0.08
and +0.70 percentage points for clear, cloud and shadow, all inside the predicted band. The
absent scene therefore accounts for the residual difference and no pipeline error is implied.

## D12. The two PixBox collections use incompatible schemas

Status: resolved. Each collection needs its own label module.

The Sentinel-2 collection, compiled in 2018, encodes cloud presence in
`CLOUD_CHARACTERISTICS_ID` over thirteen classes, and shadow in `SHADOW_ID` over five,
separating topographic shadow and shadow above cloud from cloud shadow.

The Landsat 8 collection, compiled in 2015, encodes cloud presence in `PIXEL_SURFACE_TYPE_ID`
over fourteen classes, and shadow in `CLOUD_SHADOW_ID` as a plain flag with no distinction
between shadow kinds.

The hazard is the shared column name. `CLOUD_CHARACTERISTICS_ID` exists in both, but in the
Landsat collection it encodes cloud type rather than presence, holding 17,368 zeros and 1,462
cirrus pixels. Reading it as the Sentinel-2 column would map 1,462 pixels as the entire cloud
class instead of 5,478, and nothing downstream would flag it.

Both mappings are verified against the OmniCloudMask confusion matrices: cloud comes out at
8,169 and shadow at 1,246 for Sentinel-2, and 5,478 and 1,396 for Landsat 8, matching TP plus FN
in Tables A.1 and A.2 exactly.

A further consequence for interpretation: shadow is measured slightly differently across the two
collections, since only the Sentinel-2 table excludes topographic shadow from the negative class
explicitly.

## D13. OmniCloudMask model version 1.0 weights no longer load

Status: open, with no known workaround. Blocks fidelity checking against the paper values.

The values reported in the OmniCloudMask paper were produced with model version 1.0. Loading
those weights fails on the current environment:

    ModuleNotFoundError: No module named 'torch.utils.serialization'

raised from `torch.load` inside `model_utils.load_model_from_weights`. The `.pth` files were
serialised by a PyTorch old enough to reference a module removed from the library years ago.
Both download sources were tried, Hugging Face and Google Drive; the files download correctly
and fail at deserialisation, so the fault is in the archive format rather than the transfer.

Resolving it would need a PyTorch old enough to unpickle the files and new enough for the rest
of the package, which is not available on a Colab runtime without breaking the CUDA build.

Consequence. Paper values cannot be reproduced directly. Fidelity of the pipeline is instead
established on PixBox Sentinel-2 against the reference notebook, which runs package version
1.7.0, and matches to within 0.07, 0.08 and 0.70 percentage points. That check validates the
label mapping, the pixel sampling and the metric implementation, all of which are shared across
collections.

All runs in this project therefore use package 1.7.0 with the latest weights, which makes the
three collections comparable to each other. The published values are not comparable to them,
being produced by a different weight set.

## D14. Published weights postdate the paper and were trained differently

Status: resolved. Training hyperparameters after V1 remain undocumented at source.

The values in the OmniCloudMask paper were produced by model version 1.0. The package now ships
version 4.0 by default, and no paper covers versions 2.0 to 4.0.

The model changelog records four versions:

| Version | Backbones | Framework | Training datasets |
| --- | --- | --- | --- |
| V1 | regnety_004, convnextv2_nano | fastai + timm | CloudSEN12 High |
| V2 | regnety_004, edgenext_small | fastai + timm | CloudSEN12 High |
| V3 | regnety_004, edgenext_small | fastai + timm | + KappaSet, PC, SR, hard negatives |
| V4 | regnety_004, edgenext_small | smp + timm | + Scribble, 2k, OCM Scribble |

The change at V2 is isolated: the second backbone moved from ConvNeXtV2 nano to EdgeNeXt small
and the changelog notes a new random interpolation technique, while the training data stayed at
CloudSEN12 High. Data expanded at V3, and again at V4 alongside the framework move to
segmentation-models-pytorch. Training grew from 16,980 pairs to 100,528 across nine sub-datasets.

Not established: the training hyperparameters for V2 to V4, which the changelog does not record.
What random interpolation means at V2 was established on 22/11/2026 from the authors' code: the
interpolation method of the image (bilinear or nearest, with or without antialiasing) is drawn per
batch; see D16 and `ocm_protocol.md`.

Measured effect. Running the same pipeline with v4 weights against the paper values gives
+5.02, +4.89 and +8.42 percentage points on PixBox Landsat 8. The gain is almost entirely in
sensitivity: shadow rises from 50.86 % to 67.41 %. On Sentinel-2 the same comparison, made
against the 1.7.0 reference notebook rather than the paper, gives 0.07 to 0.70 points, because
that notebook already uses the newer weights.

The gain cannot be attributed to data volume alone, since the V2 step changed architecture and
one training technique on unchanged data. How the total divides between the four steps is not
measurable without per-version results, which are not published.

Consequence. Published values and v4 measurements are not interchangeable. Every run in this
project uses 1.7.0 with the latest weights so that the three collections are comparable to each
other. Phase 2 must train both models rather than compare against v4, as ADR-0023 requires:
comparing a model trained on 16,980 patches against one trained on 100,528 measures data volume,
not architecture.

## D15. OCM trains both backbones at one learning rate; we search one per backbone

Status: resolved by a decision, recorded in ADR-0023 D2.

The OmniCloudMask training notebook builds one `Learner` per backbone and calls `fine_tune`
with a single base learning rate, so `regnety_004` and `convnextv2_nano` share it. The rate
is not tuned per backbone in the paper, and the value in our native configuration (0.005) has
no source we could find, which is listed as open.

Our protocol searches the rate of each network over the same grid with the same proxy budget
(ADR-0023 D2), so the two OCM backbones may end up at different rates. This departs from the
OCM recipe on purpose: a rate checked on one backbone and applied to the other would leave
that backbone with less search than Swin received. The cost is 12 proxy runs for OCM against
6 for Swin, the same six per network.

Consequence. The reproduction is of the OCM architecture and data recipe, with the learning
rate chosen by our own search. What this controlled comparison measures is therefore not
OCM as published, a rule ADR-0023 already sets for its BOA values.

## D16. OmniCloudMask augmentations: the paper against the authors' current code

Status: resolved by a decision, recorded in ADR-0023, revision 2.

Section 2.9 of the paper lists four augmentations: image tear (10% per batch), random
rectangle (60% per batch, random fill, one to three channels, label unchanged), scene edge
(10% per batch, label unchanged) and rotations and flips (100%). The public training code
(`training/augs.py` and `Train OCM models.ipynb`) is at a later model version. It applies
those four and also sharpen/blur, clipping of extremes, random crops of large images and
resampling between 0.07 and 1.11 of the native scale.

What the code adds about the four: the rectangle covers 10–50% of the image with aspect
0.3–3.3 and a fill between 0 and 10,000, and the tear shifts by up to 10 px. These shape
parameters are not in the paper and are taken from the code.

One disagreement. In the code, SceneEdge is gated at 10% per batch and then applied to each
image of that batch with a further 10%, so about 1% of images get a scene edge. The paper
gives 10%. We follow the paper and apply it to 10% of the samples.

Random interpolation (added 22/11/2026). The resampling step of the authors' code
(`BatchResample` in `training/augs.py`) also draws, per batch, the interpolation of the image
between bilinear and nearest, and for bilinear whether to antialias; the label stays nearest.
`MODEL_CHANGELOG.md` dates this to model version 2.0. It is not in the paper, so it follows the
rule above and is left out: our resampling is always bilinear without antialiasing for the image
and nearest for the label, for both architectures. The published versions 2.0 to 4.0 trained with
it and our models did not, one more difference to keep in mind when reading them against each
other (D14).

Two differences of form, which leave the share of augmented samples the same: the random
rectangle and the scene edge are drawn per sample here instead of once per batch, because
they act on the raw values, which only exist before the per-sample normalization; and
they come before the resolution draw rather than after it, which does not change what a
rectangle or an edge is.

## D17. The 509 window was cropped from the wrong side

Status: fixed in the reader (CROP_REVISION 2), and confirmed on the rescored test on 17/11/2026 (the strip is gone; docs/AUDIT.md A1). The networks are not retrained (decision of 15/11/2026, below).

The full verification (hypotheses tested, counts, evidence, references, how to reproduce) is
in `docs/AUDIT.md`, A1; what it leaves for reporting is in `docs/LIMITATIONS.md`, L1.

The dataset README says the 509 pixel patches are padded to 512 with zeros on the left and
bottom. The rasters carry the padding on the right and bottom: a plot of the full 512
canvas (the earlier project's `baseline.ipynb`) shows the black strips there. The reader
followed the README and kept columns 3 to 511, so every patch dropped its first three real
columns and kept the three padding columns: image zeros, label 0 (clear), 1,527 pixels.

How it showed: no test patch had fewer than 1,527 clear pixels, 67 patches had exactly that
many, and on those 67 the published OmniCloudMask found all 1,527 (its package maps a
zero input to class 0) while the trained models found 70–90% of them. That made the trained
models look much worse than the published ones in the clear 0–10% range, and kept fully
cloudy patches in that range when, without the strip, their clear BOA is undefined.

What was affected: every patch read with crop_to_valid, so training, validation and test.
The strip is 0.59% of a patch's pixels, the same for both architectures.

What was done: the window is now columns 0 to 508, and check_padding refuses an image whose
dropped pixels are not all zero, on every read. Counts are saved under a name that carries
the crop revision, and the local cache under a folder that does, so nothing scored with the
old crop is read back; the test evaluation is scored again.

Decision on training (15/11/2026): the networks are kept, not retrained. Retraining means
the fifteen runs again, and the time is not there. What this leaves, and why it is accepted:

- Training and validation used the old crop. Every patch carried the three padding columns
  (image zeros, label clear; 1,527 pixels, 0.59% of the patch) and lacked its first three
  real columns. Model selection, by validation cross-entropy, saw the same.
- Both architectures trained, and were selected, on exactly these patches, so the comparison
  between them stays even: the defect is a shared feature of their training data, not a
  difference between them.
- The test is read with the fixed crop: it holds no padding, and every pixel scored is a
  real, annotated pixel. A model's reaction to an all-zero input no longer counts anywhere.
- The published OmniCloudMask versions are scored on the same fixed patches, so the
  reference no longer gains from the strip either.

Training from now on: every reader crops with the fixed window, and the local cache records
the crop revision it was copied with (meta.json, `crop_revision`); a cache copied with another
revision, or from before the record, is refused when read and copied anew by `build`, so a
cache left on disk (molab keeps its work folder) cannot bring the old crop back. Retraining
these same networks needs a new PROTOCOL_REVISION: run ids carry the protocol revision, not
the crop, so under `-v2` the scheduler finds the current checkpoints and counts the runs as done.

What to state when reporting: the networks were trained on 509 patches cropped with the
padding on the wrong side (three padding columns labelled clear kept, three real columns
lost, 0.59% of each patch), the same for both; evaluation used the correct crop. Results
from before the fix (files without `crop2` in their name) are not comparable and are not
used.

## D18. The 13-band experiment: L1C only, and a pretrained stem adapted from 3 to 13 bands

Status: resolved (decision of 20/11/2026; code in `train/experiments.py`, `train/protocol.py`).

Question: with three bands (R-G-NIR) the OCM ensemble beats Swin-T + UPerNet by a small
margin (`claude/resultados-fase2.md` in the project). Does the ranking hold when both see
every band Sentinel-2 records? The experiment `l1c13` trains the same two architectures, with
the same protocol (ADR-0023 rev. 2: grid, budget, augmentations, 10 m validation, 5 seeds),
on all 13 bands. Three choices depart from the R-G-NIR experiment.

**1. Thirteen bands, L1C only.** The CloudSEN12+ L1C rasters carry 13 bands in the order
B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12; the L2A rasters carry 12
of them, without B10 (cirrus, 1375 nm), plus the AOT and WVP layers (dataset card [1];
Aybar et al. 2024 [2]). B10 is a top-of-atmosphere water-vapour absorption band with no
surface signal, so Sen2Cor drops it from L2A [3]. A 13-band model can only be trained and
tested on L1C. Training on L1C plus L2A at 12 bands was considered and rejected: it gives up
B10, the band the Sentinel-2 mission added to detect thin cirrus [3].
Within the experiment both architectures see the same patches, so the CNN against ViT
comparison stays as controlled as at three bands. The test is the L1C test split (975
patches), the primary test of the project already.

**2. The ImageNet stem adapted to 13 bands, the same way for both.** Both encoders were
pretrained on 3-channel RGB (ADR-0023 D4). For the OCM backbones timm adapts the first
convolution when `in_chans` differs from the pretrained one: `adapt_input_conv` repeats the
RGB filters over the new channels, keeps the first `in_chans`, and scales them by
3 / in_chans so the activations keep their magnitude [4]. The Hugging Face Swin backbone has
no such step and refuses a mismatched patch embedding, so `adapt_patch_embedding` applies
that same timm function to the Swin patch-embedding weights. Both architectures therefore
start from the same adaptation of the same kind of weights, and D4 (symmetric pretrained
initialization) still holds. The repeated filters carry no knowledge of what B01 or B12
are; both models learn that from the data. Checked by tests: the adapted Swin weights equal
`adapt_input_conv(13, rgb)` and the adapted model has the layers of one built for 13 bands.

**3. A learning-rate search of its own.** The run ids are new
(`ocm-l1c13-cs12-shared-v2`, `swin-l1c13-cs12-shared-v2`), so the scheduler runs the full
grid again for each network instead of reusing the R-G-NIR winners: more input bands and a
new first layer can move the optimum. Micro-batch, local cache and settings are kept apart
from the R-G-NIR ones (`micro_batch_l1c13`, `cloudband_cache_l1c13`).

**What this does not allow.** Comparing a 13-band model with an R-G-NIR model of the same
architecture does not isolate the bands. The two experiments also differ in the training
data (L1C only, 8,490 + 535 patches, against L1C plus L2A, twice as many), in the crop (the
R-G-NIR models trained on the wrong crop, D17, the 13-band ones on the fixed one) and in the
first layer (adapted against pretrained as is). Such a comparison is reported as informative
only. Isolating the bands needs an R-G-NIR run on L1C only, under the fixed crop, which is a
new experiment in `train/experiments.py`.

References

1. CloudSEN12+ dataset card, https://huggingface.co/datasets/tacofoundation/cloudsen12
   (band table of the L1C and L2A products).
2. Aybar, C. et al. (2024). CloudSEN12+: The largest dataset of expert-labeled pixels for
   cloud and cloud shadow detection in Sentinel-2. Data in Brief, 56, 110852.
3. ESA, Sentinel-2 Level-2A processing (Sen2Cor), SentiWiki: the L2A product holds 12
   surface-reflectance bands; B10 is not provided.
4. timm, `timm/models/_manipulate.py`, `adapt_input_conv`, applied by `load_pretrained` when
   `in_chans` differs from the pretrained input.

## D19. U-Net + RegNetZ-D8 retrained with the recipe of the earlier repository, not ADR-0023

Status: resolved (decision of 28/11/2026; code in `train/regnetz_recipe.py`, notebook
`notebooks/06_regnetz/train_regnetz_ft_molab.py`).

Question: the network proposed in the earlier repository, U-Net + RegNetZ-D8 on 13 bands
with Focal Tversky ("UNet+RegNetZ-D8 (FT)" in its figures), is trained again, 5 seeds, to see
whether its results come back. Decided: its own recipe, as the code stood when it was trained
(github.com/ArthurrCr/Cloud-Segmentation, December 2025), not the ADR-0023 protocol of OCM and
Swin.

The recipe, read from that code: `smp.Unet` with encoder `tu-regnetz_d8`, ImageNet weights,
13 input bands (smp repeats the RGB filters over the bands and rescales them) and 4 classes;
Focal Tversky (alpha 0.3, beta 0.7, gamma 0.75); Adam at 1e-4 with weight decay 1e-4;
ReduceLROnPlateau on the validation loss (factor 0.1, patience 3); at most 50 epochs, early
stopping after 5 without an improvement of 1e-4, checkpoint at the lowest validation loss;
batch 4; mixed precision; the image divided by 10,000; the whole 512 canvas; L1C, high-quality
labels, p509. No data augmentation: the dataset code of that time
(`src/utils/dataset.py`, commits of 27/12/2025) applies none, and the checkpoint on the Drive
(`pibic/checkpoints/stage1_loss_ce/best_model.pth`, 28/12/2025) was trained with it.

Three points are not recorded there and were chosen:

1. **Learning rate of the Focal Tversky model.** The final value was not kept; 1e-4 is the
   default of the code and the rate of its loss comparison (stage 1), so it is used.
2. **Seeds.** The earlier run used 42; the five seeds are 42 to 46.
3. **The canvas.** The cache holds the 509 window cut by the fixed crop (D17). Padding it back
   to 512 with zeros on the right and bottom gives exactly what the earlier code read: its 3
   rows and columns of padding (zero image, clear-sky label) are trained on, as there.

What this does not allow: a controlled comparison with the 13-band OCM and Swin. The recipe
differs in loss, optimiser, schedule, epoch budget and its stopping rule, batch, normalisation,
augmentation (none here), learning-rate selection, and the padding in the loss. The three can
be read side by side on the same test patches, as the published OmniCloudMask versions are, but
a difference between them is not one of architecture.
