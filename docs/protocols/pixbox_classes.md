# PixBox classes

Source: `cloudband/labels/pixbox.py` and `cloudband/labels/pixbox_l8.py` (real,
project code, confirmed directly — not inferred from the paper). Cross-checked
against Skakun, S. et al. (2022), Section 2.1.5 and Appendix A Table A1 (see
`cmix_protocol.md`), and Paperin, M. et al. (2021a, 2021b), Zenodo dataset
descriptions.

## Shadow is an independent attribute, not a fourth class

Both collections encode cloud and cloud shadow as two separate integer
attributes per pixel, not as mutually exclusive classes. A pixel can be
labelled both cloud and cloud shadow at once — `cloud_shadow_overlap()` counts
exactly this in both files, and the expected overlap is a known, checked
quantity (S2: 361 pixels; L8: 409 pixels). `clear` is defined as the
conjunction of both negatives (not cloud and not shadow), not as "whatever is
left over" once cloud and shadow are removed. This resolves the open question
in `cmix_protocol.md` D4: shadow was never missing from PixBox, it just isn't
a class in the same sense cloud is.

## Sentinel-2 (`labels/pixbox.py`)

| Attribute | Column | Positive IDs | Negative IDs |
| --- | --- | --- | --- |
| Cloud | `CLOUD_CHARACTERISTICS_ID` | 2, 3, 4, 5, 6, 8, 9, 10, 11, 12 | 0, 1, 7 |
| Shadow | `SHADOW_ID` | 3 | 0, 1, 2, 4 |

`TOTAL_PIXELS = 17351`, matching CMIX's and the OCM paper's own stated count.
`EXPECTED_LABEL_COUNTS = {clear: 8297, cloud: 8169, shadow: 1246}`,
`EXPECTED_CLOUD_SHADOW_OVERLAP = 361`. These four numbers are internally
consistent: 8297 + 8169 + 1246 − 361 = 17351.

One scene (`PRODUCT_ID` 872013732) is declared in the collection but its
imagery is absent from the published archive — 550 pixels cannot be scored.
See `docs/protocols/DIVERGENCES.md` D11; `drop_unavailable_scenes()` and
`unavailable_pixel_counts()` handle this explicitly rather than silently.

## Landsat 8 (`labels/pixbox_l8.py`)

Different schema from Sentinel-2, documented directly in the file's own
docstring: cloud presence lives in `PIXEL_SURFACE_TYPE_ID`, not in
`CLOUD_CHARACTERISTICS_ID` — that column exists in this collection too, but
encodes cloud type, not presence, and reading it as the Sentinel-2 column of
the same name would only capture cirrus pixels as cloud.

| Attribute | Column | Positive IDs | Negative IDs | Excluded (neither) |
| --- | --- | --- | --- | --- |
| Cloud | `PIXEL_SURFACE_TYPE_ID` | 0, 1 | 2, 3, 4, 5, 6, 7, 11, 12 | 8, 9, 10 (mixed cloud), 13 (other) |
| Shadow | `CLOUD_SHADOW_ID` | 1 | 0 | — |

Surface type IDs 8, 9, 10 (mixed cloud) and 13 (other) fall into neither the
cloud nor the not-cloud mask, so they contribute to neither `clear` nor
`cloud` — a deliberate exclusion of ambiguous pixels, not an oversight.

`TOTAL_PIXELS = 18830`, not the paper's stated 20,500 — this matches
`docs/protocols/DIVERGENCES.md` D3 exactly (Zenodo and the OCM matrices both
carry 18,830). `EXPECTED_LABEL_COUNTS = {clear: 12365, cloud: 5478,
shadow: 1396}`, `EXPECTED_CLOUD_SHADOW_OVERLAP = 409`. Consistent:
12365 + 5478 + 1396 − 409 = 18830.

## The AI4QC dataset is unrelated to what this project uses

The earlier note in this file about a Zenodo dataset ("PixBox_AI4QC", DOI
10.5281/zenodo.11121168) that reclassifies PixBox-S2 into a rasterized
four-class scheme (0-clear, 1-thick cloud, 2-thin cloud, 3-cloud shadow,
5-no-data) is real, but it is a separate product built by Bias Variance Labs
(BVLabs) for an unrelated project (AI4QC), in a different format entirely:
rasterized GeoTIFF masks over full Sentinel-2 band tiles, not the point-sampled
CSV table with `PIXEL_Y`/`PIXEL_X`/`PRODUCT_ID`/`CLOUD_CHARACTERISTICS_ID`/
`SHADOW_ID` columns that `labels/pixbox.py` reads. This project's code never
touches that dataset. There is one PixBox implementation here, not two to
choose between or report separately.

## Divergences

| Id | Divergence | Where |
| --- | --- | --- |
| D1 | CMIX Table A1 lists no shadow class for PixBox S2 or L8; PixBox encodes shadow as an independent attribute, not a class the CMIX-style binary table would show | Resolved: see `cmix_protocol.md` D1, D4, and the confirmed schema above |

## Impact on the plan

- `verify_label_counts()` in both files is exactly the "does this match the
  paper" check: it compares the loaded table's observed label counts and
  cloud/shadow overlap against the hardcoded expected values, returning an
  empty result only when everything matches. Running it against a freshly
  downloaded PixBox table is the direct way to confirm the data in hand is
  the same population the OCM paper's PixBox fidelity numbers were computed
  over, before trusting any comparison against those numbers.
- No further PixBox class research is needed for this project; the open
  items from the earlier version of this note are resolved by the code
  itself, not by further literature search.