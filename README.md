# geoloc-tr demo — Ankara

Static demo of a city-scale image geo-localization model trained on Mapillary street-level imagery.
Live at **https://baris-unver.github.io/claude/**

Predictions are **precomputed**: GitHub Pages serves static files only, so no model runs behind the
page. 48 held-out queries sweep the full error distribution — successes and failures both — from
`test_seen` (another drive down a trained street) and `test_unseen` (~1 km blocks with no ground
training data at all). To localise your own photo, run the app locally from the
[source branch](https://github.com/baris-unver/claude/tree/claude/geoloc-turkey-mapillary-qgfq5b).

Photographs © their Mapillary contributors, licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); each query links to its source
image and author. Basemaps: OpenFreeMap (vector) and Esri (raster).
