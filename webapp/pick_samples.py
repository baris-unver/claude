#!/usr/bin/env python
"""Choose the demo's pre-selected query images and write webapp/samples.json.

Picks across the ERROR QUANTILES of the full method rather than the best cases: a demo that only
shows successes misrepresents a model whose median error is ~1 km. Six from `test_seen` (a different
drive through streets the model trained on) and six from `test_unseen` (held-out ~1 km blocks with no
ground training data), both held out of training either way.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "ankara"
METHOD = "full: proto + aerial + image refine"
QUANTILES = [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]


def main():
    splits = pd.read_parquet(ROOT / "data" / "ankara" / "splits.parquet")
    out = []
    for split in ("test_seen", "test_unseen"):
        err = pd.read_parquet(RUN / f"errors_{split}.parquet")[["id", METHOD]].rename(
            columns={METHOD: "err_m"})
        df = splits[splits["split"] == split].merge(err, on="id").sort_values("err_m")
        for q in QUANTILES:
            row = df.iloc[min(int(q * len(df)), len(df) - 1)]
            out.append({
                "id": str(row["id"]),
                "split": split,
                "path": str(row["path"]),
                "gt_lat": float(row["lat"]),
                "gt_lon": float(row["lon"]),
                "recorded_err_m": float(row["err_m"]),
                "quantile": q,
                "sequence": str(row["sequence"]),
            })
    missing = [s for s in out if not Path(s["path"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} sample images are missing on disk")
    (Path(__file__).parent / "samples.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} samples")
    for s in out:
        print(f"  {s['split']:12s} q{s['quantile']:.2f}  recorded error {s['recorded_err_m']:9.1f} m")


if __name__ == "__main__":
    main()
