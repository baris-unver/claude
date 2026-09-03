# webapp — interactive tester for the trained model

    python webapp/app.py -c configs/ankara.yaml          # then open http://127.0.0.1:8000

Requires a finished pipeline (`runs/<city>/best.pt`, `cells.npz`, `image_index.npz`,
`calibration.json`) and `flask`. Runs on **CPU by default** (~0.2 s per query for ViT-S/14 at
224 px) so it does not take the GPU from a training run; `--device cuda` if you want it.

**Pre-selected queries.** `webapp/pick_samples.py` writes `samples.json`: six from `test_seen` and
six from `test_unseen`, chosen at the 5/20/40/60/80/95th percentiles of the full method's error —
deliberately spanning the distribution rather than showing only successes, because the median error
on this city is ~1 km and a best-of reel would misrepresent it. Re-run it after retraining.

**Aerial mode.** The header switch `Aerial / satellite photo` swaps in the overhead-query model
(`geoloc_tr/overhead.py`, run `scripts/09_overhead.py` first; `--overhead-config` picks its config,
default `configs/ankara_overhead.yaml`, and the switch is hidden when that run does not exist). Twelve
pre-selected ~205 m views at random headings from `webapp/pick_overhead_samples.py`, four per source at
the 10/40/70/95th error percentiles: the current Esri imagery and the two Esri **Wayback** dates
(2023-08-31, 2017-11-16) the model never trained on. Uploads: a nadir satellite / aircraft / high-drone
image of somewhere in Ankara; give **metres per pixel** if you know it, otherwise the photo is assumed
to cover ~205 m. Embeddings are averaged over 4 rotations. Entering aerial mode switches the basemap to
Satellite so the match is visible; leaving it restores the previous basemap. `?mode=overhead` deep-links it.

**Aerial mode on the static site.** `export_web.py --overhead` ships the overhead model the same way
(86 MB fp32 encoder) but compresses its 278,932-cell database to 35 MB with an uncentred PCA to 128
dims + int8 rows (measured cost ~1 point of R@100m; `proj.bin` is applied to the query in the page),
and `build_static_overhead.py` bakes the twelve pre-selected overhead queries into `site/overhead.json`
+ `site/img_overhead/`. The page loads the aerial model lazily on the first aerial upload (124 MB, once),
averages 4 rotations in one batched run (~2 s on wasm), and crops to the model's ~205 m extent when a
metres-per-pixel value is given. Checked end to end in headless Chromium against the server's
answers on the sample views.

    python webapp/export_web.py -c configs/ankara_overhead.yaml --overhead --out site/model_overhead
    python webapp/build_static_overhead.py -c configs/ankara_overhead.yaml --out site

**Upload.** Any street-level photo; there is no ground truth for it, so the panel reports the
prediction and the retrieval score but no error. It should be inside the trained city bbox for the
answer to mean anything.

**Enlarging.** Click the query image in the Result panel, or the ⤢ badge on any thumbnail, for a
full-size lightbox (Esc or click to close). ⤢ enlarges without triggering a localisation.

**Map.** MapLibre GL. Blue = prediction, green = ground truth, dashed line between them, red circles
= the top-20 candidate cells, sized and glowing by retrieval weight. `?q=<0-11>` deep-links a query,
`?style=<name>` a basemap.

**Basemaps.** `Dark` / `Positron` / `Liberty` are OpenFreeMap **vector** styles — MapLibre GL is the
same engine Mapbox GL uses, and OpenFreeMap needs no key and no account. `Canvas` (Esri Dark Gray)
and `Satellite` (Esri World Imagery, the source the aerial branch was trained on) are **raster**.

> Vector styles need a hardware WebGL rasteriser. Under software GL they download correctly and
> paint nothing — verified here: a four-line control page with nothing but MapLibre and the
> OpenFreeMap style is blank too, while the identical page with a raster style renders fine. On a
> normal GPU browser the vector styles are the better-looking option; if yours comes up blank,
> `Canvas` and `Satellite` always render.

CARTO's basemap CDN is **not** usable — it now watermarks unkeyed tiles `API KEY REQUIRED`.

**Correctness check:** live predictions reproduce the recorded evaluation errors exactly
(10.0/10.0 m, 1840.0/1840.0 m, 48.7/48.7 m on three spot checks), i.e. the app's inference path is
the same one `07_evaluate.py` measured.
