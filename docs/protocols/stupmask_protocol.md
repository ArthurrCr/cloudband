# STUPmask protocol

Source: Pang, S. et al. (2026). Enhancing cloud detection across multiple satellite sensors
using a combined Swin Transformer and UPerNet deep learning model. Remote Sensing of
Environment 334, 115206.

Code and weights: not publicly available. The data availability section lists dataset links
only. Checked on GitHub, Zenodo and HuggingFace. The model has to be retrained.

## The model is binary

Figure 2 labels the output a binary cloud mask and Eq. 6 is two-class balanced accuracy. There
is no shadow class anywhere in the paper: not in training, not in the metrics, not in the
figures.

Two consequences:

1. Of the three experiments, native STUPmask only participates in cloud. Shadow is impossible,
   and since the model is strictly binary, clear is the same problem as cloud with TP and TN
   swapped, giving an identical BOA. It yields one independent number, not three.
2. Native STUPmask does not solve the same task as OCM. This is a second reason, beyond channel
   count, why project note 2.2 keeps the two configurations separate.

Comparison against a published STUPmask number requires binarising CloudSEN12+ as
cloud = {thick, thin} and not cloud = {clear, shadow}. Figure 10 is labelled "all clouds", which
supports including thin cloud in the positive class.

## Contradiction over the Sentinel-2 training set

| Section | Claim |
| --- | --- |
| 2.2 | CloudSEN12 (8,942 patches) is training; CMC (2,052 patches) and CloudSEN12 validation (975) are independent validation |
| 3.2 | A single model trained on L8 Biome and CMC, with no mention of CloudSEN12 |

The two readings are incompatible. Which one holds decides whether the most convenient fidelity
target, BOA 93.64 % on the CloudSEN12 test split, came from a model that had seen CloudSEN12
during training.

Working assumption: the section 2.2 reading, which is more detailed and carries patch counts.
See `docs/protocols/DIVERGENCES.md` D1.

## Bands

Landsat 8, section 3.2, literal: blue, green, red, NIR, SWIR, cirrus and BT channels, giving
B2, B3, B4, B5, B6, B7, B9, B10, B11, nine channels.

Sentinel-2: the paper only states that the features span blue to SWIR with the TIRS channel
excluded. The list is not enumerated. Two readings:

| Reading | Bands | Count |
| --- | --- | --- |
| A, mirroring Landsat 8 | B02, B03, B04, B08, B10, B11, B12 | 7 |
| B, literal "blue to SWIR" | B02 to B12 | 11 |

Reading A is adopted. It is an inference by analogy with the Landsat list, not a literal
statement of section 3.2. See D4.

Which NIR is unstated. Table 1 of the same paper classifies B08 as NIR and B8A as red edge,
which favours B08 and indicates that STUPmask and OCM use different NIR bands.

## Preprocessing

| Item | Status | Value |
| --- | --- | --- |
| Processing level | stated | TOA reflectance for VIS-SWIR plus brightness temperature for TIRS, confirming L1C |
| Standardisation | inferred | Mentioned, method unspecified |
| Sentinel-2 resize | stated | 509x509 to 512x512 by bilinear interpolation |
| Landsat patches | stated | 512 px with 24 px overlap, 10,230 training and 9,986 validation |
| Landsat split | stated | 48 of the 96 Biome scenes for training by stratified sampling, 48 for validation |

The bilinear resize differs from OCM, which stays at 509 native, and from project note 2.11.
These are different stages rather than a conflict, but `stup-*-native` and `swin-*-shared`
therefore have different preprocessing by design.

## Training, roughly 30 % specified

| Item | Status | Value |
| --- | --- | --- |
| Optimiser | stated | AdamW, beta1 0.9, beta2 0.999, weight decay 0.01 |
| Schedule | stated | Warmup to a maximum, then decay |
| Loss | stated | Cross entropy with feature fusion at each UPerNet layer |
| Encoder | stated | Swin Transformer, blocks 2, 2, 6, 2 per Figure 2 |
| Decoder | stated | UPerNet with PPM pooling 1, 2, 3, 6 and lateral connections |
| Swin variant | absent | The 2, 2, 6, 2 depth matches Swin-T, but this is read off a figure |
| Learning rate | absent | |
| Batch size | absent | |
| Epochs | absent | |
| Warmup length | absent | |
| Initialisation | absent | |
| Augmentation | absent | None mentioned |
| Window size | absent | |
| Seed | absent | |

Reproduction requires independent choices for learning rate, batch size, epochs and
initialisation. This is a limitation of the source and is stated as such in
`docs/LIMITATIONS.md`. See ADR-0023 for the learning rate search protocol.

## Metrics

    OA  = (TP + TN) / (TP + TN + FP + FN)
    UA  = TP / (TP + FP)
    PA  = TP / (TP + FN)
    BOA = 0.5 * ( PA + TN / (TN + FP) )
    F1  = 2 * UA * PA / (UA + PA)
    IoU = TP / (TP + FP + FN)
    CA  = cloud pixels / valid pixels
    CAD = difference in CA between prediction and reference

Identical to OCM. The text below the equations swaps TP and TN for FP and FN. See D6.

## Fidelity targets

| Dataset | Sensor | OA | BOA |
| --- | --- | --- | --- |
| L8 Biome, 48 validation scenes | L8 | 97.51 | 96.91 |
| S2 CMC, 2,052 patches | S2 | 96.27 | 96.23 |
| CloudSEN12 test | S2 | | 93.64 |
| PixBox S2, 17,351 px | S2 | 91.36 | 91.43 |
| PixBox L8, 20,500 px | L8 | | 91.33 |
| CESBIO, 38 scenes | S2 | 96.42 | 94.37 |
| SPARCS, 80 scenes | L8 | 94.09 | 94.39 |

Cloud amount agreement: R-squared 0.98 for Landsat 8 and 0.99 for Sentinel-2, with cloud amount
differences of 0.27 % and -0.81 %. Worst case is snow and ice on Landsat 8 at BOA 91.63 %.

CloudSEN12 test at BOA 93.64 % is the only target available without a large download. Subject
to D1 and D5.

## Divergences

| Id | Divergence | Where |
| --- | --- | --- |
| D1 | CloudSEN12 is training (2.2) or is not (3.2) | Internal |
| D2 | CloudSEN12 validation: 975 patches (2.2) versus 963 scenes (4.4.1) | Internal |
| D3 | PixBox L8: 20,500 px (4.4.1) versus 18,830 in Zenodo and in the OCM matrices | Paper vs source |
| D4 | The Sentinel-2 band list is not enumerated, yet project note 2.2 cites section 3.2 as literal | Notes vs paper |
| D5 | The paper uses CloudSEN12, this project uses CloudSEN12+, which are different sets | Paper vs project |
| D6 | TP and TN swapped with FP and FN in the text below the equations | Internal |
| D7 | CESBIO spelled CEOBIO | Internal |

Full entries in `docs/protocols/DIVERGENCES.md`.

## Impact on the plan

- STUPmask contributes to the cloud experiment only; shadow is impossible.
- The model must be trained from scratch, with roughly 70 % of hyperparameters chosen
  independently of the paper.
- The Sentinel-2 band set is an inference between two readings, recorded as a project decision.
- Preprocessing differs from the shared phase 2 protocol by design.

Sequencing: phase 0 completes with OCM before STUPmask begins. OCM has published weights, a
documented protocol and clean fidelity targets. STUPmask needs training, carries an internal
contradiction about its training set, and leaves most hyperparameters undefined.