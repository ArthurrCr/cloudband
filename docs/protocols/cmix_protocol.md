# CMIX protocol

Source: Skakun, S. et al. (2022). Cloud Mask Intercomparison eXercise (CMIX): An
evaluation of cloud masking algorithms for Landsat 8 and Sentinel-2. Remote Sensing
of Environment 274, 112990. Open access, CC BY 4.0. DOI 10.1016/j.rse.2022.112990.

## Scope

CMIX intercompares 10 cloud masking algorithms for Landsat 8 and Sentinel-2 against
5 existing reference datasets. It is an evaluation exercise and a metrics protocol,
not a cloud detection algorithm and not a training dataset.

## The protocol is binary cloud/non-cloud only — this project extends it

Literal, Section 1: "Within CMIX, a qualitative definition of 'cloud' is adopted...
ultimately all data are converted to 'cloud' and 'non-cloud' classes to perform a
consistent intercomparison... Cloud shadows are not considered in this study, since
it is typically a cloud-derived product, and its performance heavily depends on
accuracy of cloud detection. Consequently, efforts are primarily directed to cloud
mask evaluation."

This project's `eval/experiments.py` runs three independent one-vs-rest binary
experiments — clear, cloud, shadow. CMIX itself runs one — cloud versus non-cloud.
The metric formulas below are taken directly from CMIX and applied unchanged to all
three; the three-experiment structure is this project's own extension, not
something the CMIX paper itself does. See D1.

## Metrics (Section 2.3, Table 4 and Table 5)

Confusion matrix layout, Table 4:

|              | Reference cloud     | Reference non-cloud  |
| ------------ | -------------------- | -------------------- |
| Map cloud    | n_cloud_as_cloud     | n_ncloud_as_cloud    |
| Map non-cloud| n_cloud_as_ncloud    | n_ncloud_as_ncloud   |

Metric equations, Table 5:

    OA  = (n_cloud_as_cloud + n_ncloud_as_ncloud) / total
    BOA = 0.5 * ( PA + n_ncloud_as_ncloud / (n_ncloud_as_cloud + n_ncloud_as_ncloud) )
    PA  = n_cloud_as_cloud / (n_cloud_as_cloud + n_cloud_as_ncloud)
    UA  = n_cloud_as_cloud / (n_cloud_as_cloud + n_ncloud_as_cloud)

BOA cites Brodersen et al. (2010), "The balanced accuracy and its posterior
distribution."

These match `eval/metrics.py` exactly: PA = producer_accuracy (sensitivity),
UA = user_accuracy (precision), OA = overall_accuracy, BOA =
balanced_overall_accuracy. F1 and IoU, also in `eval/metrics.py`, are not part of
CMIX's own metric set — Table 5 has only OA, BOA, PA and UA. They are this
project's own addition. See D2.

"Performance metrics were estimated from confusion matrices that incorporated all
valid pixels over all scenes available in the dataset" — CMIX's own reported
numbers pool counts across every scene in a dataset before computing a metric, not
an average of per-scene metrics. This matches this project's `pooled()` function.
The per-scene metric table this project computes for the paired Wilcoxon test
(note 2.10) is not itself a CMIX quantity: it applies CMIX's same formulas per
scene instead of pooled across scenes, because a paired test needs one value per
scene, not one value per dataset. See D3.

## Reference datasets (Section 2.1, Table 1)

| Dataset  | Domain                  | Classes | Sensors | Resolution         | Scenes            | Availability |
| -------- | ------------------------ | ------- | ------- | ------------------- | ------------------ | ------------ |
| CESBIO   | Fully classified scenes  | 6       | S2      | 60 m                | 30                 | zenodo.org/record/1460961 |
| GSFC     | Sample polygons          | 4       | L8, S2  | polygons            | L8: 6, S2: 28      | doi.org/10.17632/r7tnvx7d9g.1 |
| Hollstein| Sample polygons          | 6       | S2      | polygons, 20 m      | 59                 | git.gfz-potsdam.de/EnMAP/sentinel2_manual_classification_clouds |
| L8Biome  | Fully classified scenes  | 4       | L8      | 30 m                | 96                 | doi.org/10.5066/F7251GDH |
| PixBox   | Sample pixels            | 10      | S2, L8  | S2: 10 m, L8: 30 m  | S2: 29, L8: 11     | zenodo.org/record/5036991, zenodo.org/record/5040271 |

PixBox original sources:
Paperin, M., Wevers, J., Stelzer, K., Brockmann, C. (2021a). PixBox Sentinel-2
pixel collection for CMIX (v1.0). Zenodo. https://doi.org/10.5281/zenodo.5036991.
Paperin, M., Stelzer, K., Lebreton, C., Brockmann, C., Wevers, J. (2021b). PixBox
Landsat 8 pixel collection for CMIX (v1.0). Zenodo.
https://doi.org/10.5281/zenodo.5040271.

S2: 17,351 pixels at 10 m, from 29 Sentinel-2 A/B L1C products, TOA reflectance.
L8: 20,500 pixels at 30 m, from 11 Landsat-8 L1 products, TOA reflectance.

These pixel and scene counts match `ocm_protocol.md`'s own fidelity targets
exactly. This document confirms they trace back to PixBox's own published counts,
not something the OCM paper invented independently.

## PixBox cloud/non-cloud binarization used by CMIX (Appendix A, Table A1)

CMIX's own Table A1 gives the exact mapping it used to reduce each dataset's
native classes to a binary cloud/non-cloud label, for CMIX's own binary analysis:

| Dataset  | Cloud                                                                                                    | Non-cloud |
| -------- | --------------------------------------------------------------------------------------------------------- | --------- |
| CESBIO   | Low clouds, high clouds                                                                                    | Shadow, land, water, snow |
| GSFC     | Cloud, thin cloud                                                                                           | Clear, cloud shadow |
| Hollstein| Cloud, cirrus                                                                                               | Clear, water, shadow, snow |
| L8Biome  | Thin cloud, thick cloud                                                                                     | Shadow, clear |
| PixBox S2| Opaque, thick semi-transparent cloud, average density semi-transparent cloud, semi-transparent cloud, thin semi-transparent cloud, fog, haze | Clear |
| PixBox L8| Cloud, semi-transparent cloud                                                                               | Clear land, clear snow/ice, clear water, mixed snow_ice/water |

This table shows only the reduction CMIX performs for its own binary analysis —
it is not the full native PixBox class list, and it has no "shadow" row for
either PixBox dataset, consistent with CMIX never analyzing shadow at all (D1).
It is not evidence that PixBox itself lacks a shadow class. See `pixbox_classes.md`
and D4.

## PixBox sampling design (Section 2.1.5, 2.1.6)

"PixBox dataset was sampled in such a way, so non-challenging (e.g., opaque thick
clouds) and challenging (e.g., semi-transparent clouds, cloud boundaries) cases
are equally present in the dataset" — unlike CESBIO/L8Biome (full scenes labelled)
or Hollstein/GSFC (homogeneous polygons), PixBox deliberately oversamples hard
cases relative to their true area proportion. Across CMIX's own four Sentinel-2
datasets, average algorithm BOA was consistently lowest on PixBox (80.0 ± 5.3%,
against 85.9-89.4% for the other three) because of this deliberate
difficulty-balancing, not because PixBox is lower quality. PixBox L8 "has a strong
spatial focus on the Northern European coastal areas" and both PixBox sets skew
toward water surface (32% for S2, 60% for L8), unlike PixBox S2, which the paper
states is otherwise "spatially, temporally, and thematically evenly distributed."

## Divergences

| Id | Divergence | Where |
| --- | --- | --- |
| D1 | CMIX's own protocol is cloud/non-cloud binary only, explicitly excluding shadow (Section 1); this project runs three independent one-vs-rest experiments using CMIX's metric formulas, which is this project's extension, not CMIX's own scope | Paper vs project |
| D2 | CMIX's Table 5 defines OA, BOA, PA, UA only; F1 and IoU in `eval/metrics.py` are this project's addition | Paper vs project |
| D3 | CMIX's own reported numbers pool counts across every scene in a dataset (Section 2.3); this project's per-scene metric table, feeding the Wilcoxon test, applies the same formulas per scene instead, which CMIX itself never reports | Paper vs project |
| D4 | Table A1 shows no shadow row for either PixBox dataset, because CMIX excludes shadow from its own analysis entirely (D1), not because PixBox lacks a shadow class; both PixBox Zenodo records state shadow is one of PixBox's own categories | Internal, resolved by the dataset's own description |

## Impact on the plan

- The OA/BOA/PA/UA formulas already coded in `eval/metrics.py` are confirmed to
  match CMIX's Table 5 exactly, checked against the primary source, not assumed.
- What this project has been calling "the CMIX protocol" throughout is precisely
  CMIX's metric formulas applied per class to three one-vs-rest experiments — not
  CMIX's own binary-only, shadow-excluded scope. Worth stating this distinction
  explicitly in the methodology section, since a reader who knows the actual CMIX
  paper would otherwise expect one binary result, not three.
- PixBox's deliberate hard-case oversampling is a citable, documented explanation
  for why PixBox scores are consistently the lowest across reference datasets in
  this project's own fidelity tables — an expected property of the sampling
  design, not a red flag about data quality.