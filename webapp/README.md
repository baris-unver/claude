# webapp — interactive tester for the trained model

    python webapp/app.py -c configs/ankara.yaml          # then open http://127.0.0.1:8000

Requires a finished pipeline (`runs/<city>/best.pt`, `cells.npz`, `image_index.npz`,
`calibration.json`) and `flask`. Runs on **CPU by default** (~0.2 s per query for ViT-S/14 at
224 px) so it does not take the GPU from a training run; `--device cuda` if you want it.

**Pre-selected queries.** `webapp/pick_samples.py` writes `samples.json`: six from `test_seen` and
six from `test_unseen`, chosen at the 5/20/40/60/80/95th percentiles of the full method's error —
deliberately spanning the distribution rather than showing only successes, because the median error
on this city is ~1 km and a best-of reel would misrepresent it. Re-run it after retraining.

**Upload.** Any street-level photo; there is no ground truth for it, so the panel reports the
prediction and the retrieval score but no error. It should be inside the trained city bbox for the
answer to mean anything.

**Map.** Blue = prediction, green = ground truth, dashed line between them, red circles = the top-20
candidate cells sized by retrieval weight. `?q=<0-11>` deep-links a pre-selected query.

**Correctness check:** live predictions reproduce the recorded evaluation errors exactly
(10.0/10.0 m, 1840.0/1840.0 m, 48.7/48.7 m on three spot checks), i.e. the app's inference path is
the same one `07_evaluate.py` measured.
