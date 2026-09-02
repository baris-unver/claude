# geoloc-tr demo — Ankara

Image geo-localization for a Turkish city, trained on Mapillary street-level imagery.
Live at **https://baris-unver.github.io/claude/**

**Upload works.** GitHub Pages cannot run a model, so the model is shipped to your browser instead
and runs there via onnxruntime-web — the photo never leaves your machine. The encoder downloads on
your first upload (86 MB, then cached) and a query takes well under a second.

The 48 pre-selected queries are held-out images sampled across the *whole* error distribution of
both splits — successes and failures — with answers precomputed server-side using the full method.
Uploads use the in-page pipeline (prototypes + aerial + refine), which the evaluation put about one
point of R@100m below the full method, because the image-refine step needs a 97,834-image index that
is too large to ship. The port was checked against the server: predictions agree to within 19 m on a
model whose median error is 775 m.

Photographs © their Mapillary contributors, licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); each query links to its source
image and author. Basemaps: OpenFreeMap (vector) and Esri (raster).
[Source](https://github.com/baris-unver/claude/tree/claude/geoloc-turkey-mapillary-qgfq5b).
