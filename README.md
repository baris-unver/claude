# geoloc-tr — city-scale image geo-localization for Türkiye from Mapillary

A from-scratch re-implementation of the recipe in
**"Scaling Image Geo-Localization to Continent Level"** (Lindenberger, Sarlin, Hosang, Balice, Pollefeys,
Lynen, Trulls; NeurIPS 2025, [arXiv 2510.26795](https://arxiv.org/abs/2510.26795),
[project page](https://scaling-geoloc.github.io/)), scaled down from a continent to **one Turkish city**
and using **Mapillary** street-level imagery (open, API-accessible) instead of Google Street View.

Default city is **Ankara** (`configs/ankara.yaml`); İstanbul, İzmir, Bursa and Antalya presets are included and
any bbox works (`data.bbox: [west, south, east, north]`).

## What the paper does, and what this repo does

| Paper (continent) | This repo (city) |
|---|---|
| Ground images partitioned into S2 cells at several levels; a ViT is trained with a **proxy classification** task over the cells | Same: hierarchical S2 heads (default levels 11/13/15/17 ≈ 4.5 km / 1.1 km / 280 m / 70 m), cosine classifiers, distance-smoothed cross-entropy (`geoloc_tr/losses.py`) |
| Classifier weights are the **prototypes**; they are extracted and **upsampled to the target resolution through the S2 hierarchy** | `geoloc_tr/database.py`: finest-level prototypes → every child cell at `cells.database_level` (default 18 ≈ 35 m) inherits its parent's prototype |
| **Aerial tiles** covering each cell are encoded and concatenated with the prototypes; both are combined with a **calibration factor** | Aerial encoder trained with a ground↔aerial InfoNCE term; per-cell aerial codes; `alpha` chosen on the validation split by maximising recall@100 m (`evaluate.calibrate_alpha`) |
| Query embedding is matched against the cell database by cosine similarity; a refinement step gives the final position | `geoloc_tr/localize.py`: top-k retrieval → score-weighted centroid of the top-k cells near the best one → optional snap to the most similar training images inside that neighbourhood |
| Evaluated by recall within 100 m / 200 m / … over 433 000 km² | Recall @ 25/50/100/200/500/1000/5000 m on held-out **sequences** (seen areas) and held-out **spatial blocks** (unseen areas) |

The main scientific point of the paper — classification as a proxy that yields retrieval-quality
prototypes, plus aerial codes to cover places with no ground imagery — is preserved. What is scaled
down is data (one city, ~10⁴–10⁶ Mapillary images rather than ~10⁸ Street View images) and the backbone
(DINOv2 ViT-S/14 by default; ViT-B/L are one config line away).

## Layout

```
geoloc_tr/
  config.py     city presets, dataclass config, YAML + CLI overrides
  geo.py        haversine, S2 cell ids / centres / children, CellHierarchy, prototype upsampling
  mapillary.py  coverage vector tiles (z=14), Graph API metadata, thumbnail download, pano crops
  aerial.py     XYZ tile cache + per-cell patches (Esri World Imagery by default, or Mapbox/own server)
  splits.py     block-level + sequence-level train/val/test splits, sequence thinning
  data.py       datasets / transforms
  model.py      backbone (timm DINOv2 or tiny CNN), projection, cosine heads, aerial encoder, checkpoints
  losses.py     smoothed hierarchical CE, ground↔aerial InfoNCE
  train.py      training loop with per-epoch prototype-retrieval validation
  database.py   cell database (prototypes upsampled + aerial codes) and training-image index
  localize.py   retrieval + refinement
  evaluate.py   recall@d, calibration, result tables
  synthetic.py  offline synthetic "city" used by the tests
scripts/        01..08 pipeline steps + make_synthetic.py
configs/        ankara / istanbul / izmir / bursa / antalya / synthetic
tests/          unit tests + an offline end-to-end run
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # torch/torchvision: install the CUDA build you need first if on GPU
cp .env.example .env           # put your Mapillary client token in MAPILLARY_TOKEN
```

Mapillary token: register an application at https://www.mapillary.com/dashboard/developers and copy its
*Client Token* (looks like `MLY|123…|abc…`). The token is passed as `access_token` to the Graph API and the
vector-tile endpoint.

## Run the pipeline for a city

```bash
C=configs/ankara.yaml
python scripts/01_fetch_coverage.py -c $C           # coverage tiles -> data/ankara/coverage.parquet
python scripts/02_download_images.py -c $C          # Graph API metadata + thumbnails -> metadata.parquet, images/
python scripts/03_make_splits.py -c $C              # train / val / test_seen / test_unseen
python scripts/04_fetch_aerial.py -c $C             # aerial tiles + one patch per database cell (optional)
python scripts/05_train.py -c $C                    # runs/ankara/{best,last}.pt, train_log.csv
python scripts/06_build_database.py -c $C           # cells.npz, image_index.npz, calibration.json
python scripts/07_evaluate.py -c $C                 # runs/ankara/results.md
python scripts/08_predict.py -c $C my_photo.jpg --map out.html
```

Every script accepts `-s key=value` overrides, e.g. `-s data.city=izmir -s train.epochs=30`, or
`-s data.max_images=20000` for a quick first pass.

### Step details and knobs

* **01 coverage** – reads the `image` layer of the z=14 coverage vector tiles
  (`tiles.mapillary.com/maps/vtp/mly1_public/2/14/x/y`), which lists every image with id, sequence,
  capture time, heading and `is_pano`. This avoids the bbox-search endpoint's size limit. `--method bbox`
  uses the Graph API bbox search instead.
* **02 download** – batch Graph API requests (`/?ids=…&fields=…`) for the SfM-corrected
  `computed_geometry`, headings and the `thumb_1024_url`; downloads and resizes to `image_max_side`.
  Panoramas are skipped by default; `data.include_pano: true` slices them into `pano_crops` views.
* **03 splits** – `split.unseen_block_frac` of the S2 level-13 blocks (~1 km) are held out entirely
  (`test_unseen`); inside the rest, whole sequences go to `val`/`test_seen`. Train sequences are thinned
  to `min_train_spacing_m` between frames, eval sequences to `split.eval_spacing_m`.
* **04 aerial** – for every database cell (children of the finest class cells at level 18) a
  `aerial.patch_px` square is cut from zoom-`aerial.zoom` tiles (z18 ≈ 0.46 m/px at 40°N, so 224 px ≈ 100 m
  of context). The default source is Esri World Imagery; check its terms for your use and swap
  `aerial.url_template` for Mapbox (`…/{z}/{x}/{y}?access_token={token}` with `MAPBOX_TOKEN`) or your own tiles.
  Set `aerial.enabled: false` to run ground-only.
* **05 train** – DINOv2 ViT-S/14 via `timm` (weights come from the Hugging Face hub on first use),
  first `freeze_backbone_blocks` blocks frozen, AdamW + cosine schedule, AMP. Loss = Σ levels
  smoothed-CE + `aerial_loss_weight` · InfoNCE. Validation each epoch uses prototype retrieval on the
  val split; `best.pt` maximises val recall@100 m.
* **06 database** – prototypes upsampled to level 18, aerial codes per cell, the training-image index,
  and the aerial weight `alpha` picked on `val`.
* **07 evaluate** – prints and stores a table like the one below for `test_seen` and `test_unseen`, with
  baselines: finest-level classification, plain nearest-image retrieval, prototypes only, + refinement,
  + aerial, full.

## Offline smoke test (no token, no GPU, ~1 min)

```bash
python -m pytest -q                      # unit tests + end-to-end synthetic run
# or the same through the scripts:
C=configs/synthetic.yaml
python scripts/make_synthetic.py -c $C && python scripts/03_make_splits.py -c $C && \
python scripts/04_fetch_aerial.py -c $C && python scripts/05_train.py -c $C --no-pretrained && \
python scripts/06_build_database.py -c $C && python scripts/07_evaluate.py -c $C
```

The synthetic city renders images whose appearance is a deterministic function of position (and
weakly of heading) inside a 2.5 × 2.2 km box of Ankara coordinates, with synthetic aerial tiles. A
tiny CNN trained for 8 epochs on CPU gives, on held-out sequences:

| method | R@25m | R@50m | R@100m | R@200m | R@500m | median m |
|---|---|---|---|---|---|---|
| classification @ L17 (cell centre) | 63.0 | 93.9 | 94.7 | 94.7 | 99.6 | 19.5 |
| prototypes, top-1 cell (L18) | 26.8 | 83.3 | 94.7 | 94.7 | 99.2 | 33.8 |
| prototypes + top-k refine | 65.0 | 93.9 | 94.7 | 94.7 | 99.6 | 21.2 |
| prototypes + aerial (α=0.25) + refine | 78.0 | 94.3 | 94.7 | 94.7 | 100.0 | 17.4 |
| full: proto + aerial + image refine | 94.3 | 94.7 | 94.7 | 94.7 | 100.0 | 3.9 |

and ~20 % @100 m on held-out blocks (no ground training data there). The synthetic aerial tiles carry
no information about the ground renders, so the aerial branch cannot help there and calibration keeps
`alpha` small; with real imagery the aerial branch is what should lift the unseen-block numbers. These numbers only demonstrate that
the pipeline is wired correctly; they say nothing about real-world accuracy.

## Expectations for a real city

* **Data volume.** Mapillary coverage in Turkish cities is uneven: İstanbul has by far the most,
  Ankara/İzmir/Bursa/Antalya have decent main-road coverage, mostly dash-cam sequences. Expect
  10⁴–10⁶ images per city bbox; step 01 prints the count before anything is downloaded, so adjust the
  bbox or `data.max_images` from there. Thumbnail download at 1024 px is ~100–150 KB/image.
* **Compute.** ViT-S/14 at 224 px, batch 128: ~1.5 min/epoch per 100 k images on one modern GPU;
  20 epochs on 300 k images ≈ 1.5 h. The aerial encoder doubles the forward cost.
* **What to look at first.** `test_seen` recall@100 m is the headline number (a different drive through
  a known street). `test_unseen` shows how far aerial codes take you where no ground imagery was seen.
  `runs/<city>/errors_<split>.parquet` has per-query errors for maps/failure analysis.
* **Levers.** More epochs and ViT-B (`model.backbone: timm:vit_base_patch14_dinov2.lvd142m`), a larger
  `min_train_spacing_m` when sequences are very dense, `cells.min_images_per_class` to control class
  count, `retrieval.refine_radius_m` and `top_k` for the refinement, and `include_pano` if your city has
  many 360° captures.

## Assumptions and caveats

* City choice: the request said "a city in Türkiye … use another one"; I read that as *not İstanbul*,
  so Ankara is the default. Switching is one config line.
* Mapillary positions are consumer-GPS (SfM-corrected where available via `computed_geometry`); a few
  metres of label noise is inherent, so recall@25 m is soft.
* Network access to Mapillary and tile servers was not available in the environment this was written
  in, so the download code follows the documented v4 endpoints but was not exercised against the live
  API; everything after download is verified offline by the tests.
* Aerial imagery terms of service are the user's responsibility; the default Esri endpoint is used only
  as a convenient example.
