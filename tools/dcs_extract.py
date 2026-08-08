#!/usr/bin/env python3
"""
dcs_extract.py — convert a SmartPilot DCS scenario bundle into a single
binary replay file for the vio_hold replay harness (tools/replay_dcs.c).

Input:  an extracted bundle directory (the one containing manifest.json,
        flights/0001/...). Frames are 1920x1080 JPEG; ownship telemetry is
        100 Hz JSONL with body-frame gyro rates.

Output layout (little-endian), one file:

  header:
    magic     u32   0x56484452  ("VHDR")
    version   u32   1
    w, h      u16 x2             frame size (VH_IMG_W x VH_IMG_H)
    fx,fy,cx,cy f32 x4           intrinsics at output resolution
    tilt_deg  f32                camera tilt about body right axis (+up)
    n_gyro    u32
    n_frames  u32
  gyro records (n_gyro):   t_us u64, wx,wy,wz f32   (body rad/s, DCS native
                                                     x fwd, y up, z right)
  frame records (n_frames): t_us u64, w*h bytes Y8

The 1920x1080 render has hfov 110 deg / vfov 78 deg (ideal pinhole,
fx ~= 672.3, fy ~= 666.9). We center-crop to 4:3 (1440x1080) and scale to
the target size, so fx' = fx * out_w/1440, fy' = fy * out_h/1080.
"""
import argparse, glob, json, math, os, struct, sys
import numpy as np
from PIL import Image

FULL_W, FULL_H = 1920, 1080

def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle_dir", help="extracted bundle dir (contains manifest.json)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--flight", default="0001")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--tshift-ms", type=float, default=0.0,
                    help="extra shift added to frame timestamps (ms), for "
                         "experiments. Normal ingestion needs none: the "
                         "archive's lag-corrected observation clock is used "
                         "automatically when present.")
    args = ap.parse_args()

    root = args.bundle_dir
    fdir = os.path.join(root, "flights", args.flight)

    # --- camera model from scenario metadata ---
    scen = json.load(open(os.path.join(root, "scenario", "scenario.json")))
    cal = scen.get("metadata", {}).get("camera_calibration", {})
    hfov = cal.get("hfov_deg") or scen["acquisition"]["camera_hfov_deg"]
    vfov = cal.get("vfov_deg") or scen["acquisition"]["camera_vfov_deg"]
    tilt = cal.get("tilt_deg", scen.get("acquisition", {}).get("camera_tilt_deg", 0.0))
    fx_full = (FULL_W / 2) / math.tan(math.radians(hfov) / 2)
    fy_full = (FULL_H / 2) / math.tan(math.radians(vfov) / 2)

    crop_w = FULL_H * args.width // args.height       # 4:3 -> 1440
    fx = fx_full * args.width / crop_w
    fy = fy_full * args.height / FULL_H
    cx, cy = args.width / 2.0, args.height / 2.0

    # --- gyro: body-frame rad/s from native_dcs_xyz_dps ---
    gyro = []
    for r in load_jsonl(os.path.join(fdir, "logs", "ownship_telemetry_100hz.jsonl")):
        if r.get("kind") != "ownship_telemetry":
            continue
        g = r["ownship"]["imu"]["gyroscope"]
        if not g.get("valid", False):
            continue
        t_us = r["recv_monotonic_ns"] // 1000
        w = [math.radians(v) for v in g["native_dcs_xyz_dps"]]
        gyro.append((t_us, *w))
    gyro.sort(key=lambda s: s[0])
    # drop duplicate timestamps (keep first)
    dedup, last_t = [], None
    for s in gyro:
        if s[0] != last_t:
            dedup.append(s)
            last_t = s[0]
    gyro = dedup

    # --- frames ---
    # Frame time selection (archive contract, schema v2 review 2026-08-08):
    #  1. observation_monotonic_ns — the lag-corrected clock of the DCS state
    #     actually visible in the pixels. Present in schema v2 (hoverheli_r9+),
    #     where capture_monotonic_ns is the RAW x11 capture-completion time.
    #  2. schema v2 without an observation field: capture - camera_pipeline_lag.
    #  3. legacy schema (no observation field, no capture_clock_source):
    #     capture_monotonic_ns is already lag-corrected per the v1 policy —
    #     use it directly. Never guess a constant lag.
    frames = []
    used_observation = False
    for r in load_jsonl(os.path.join(fdir, "logs", "frame_index.jsonl")):
        if r.get("kind") != "frame":
            continue
        timing = r.get("camera_timing") or {}
        obs = timing.get("observation_monotonic_ns",
                         r.get("camera_observation_monotonic_ns"))
        if obs is not None:
            t_ns = int(obs)
            used_observation = True
        else:
            cap = timing.get("capture_monotonic_ns",
                             r.get("capture_monotonic_ns"))
            if cap is None:
                sys.exit("frame record lacks any camera timestamp")
            t_ns = int(cap)
            if timing.get("capture_clock_source") == "x11_capture_completion":
                lag_ms = timing.get("camera_pipeline_lag_ms")
                if lag_ms is None:
                    sys.exit("raw capture clock without a recorded pipeline "
                             "lag; refusing to guess")
                t_ns -= int(round(float(lag_ms) * 1e6))
        t_us = t_ns // 1000
        t_us = max(0, t_us + int(args.tshift_ms * 1000))
        frames.append((t_us, os.path.join(root, r["file"])))
    if used_observation and args.tshift_ms:
        print("WARNING: observation clock in use AND --tshift-ms given — "
              "make sure you are not correcting the pipeline lag twice",
              file=sys.stderr)
    frames.sort(key=lambda s: s[0])

    if not frames:
        sys.exit("no frames found")

    x0 = (FULL_W - crop_w) // 2
    with open(args.out, "wb") as out:
        out.write(struct.pack("<IIHH4ffII", 0x56484452, 1,
                              args.width, args.height,
                              fx, fy, cx, cy, float(tilt),
                              len(gyro), len(frames)))
        for t_us, wx, wy, wz in gyro:
            out.write(struct.pack("<Qfff", t_us, wx, wy, wz))
        for t_us, path in frames:
            img = Image.open(path).convert("L")
            if img.size != (FULL_W, FULL_H):
                sys.exit(f"unexpected frame size {img.size} in {path}")
            img = img.crop((x0, 0, x0 + crop_w, FULL_H))
            img = img.resize((args.width, args.height), Image.LANCZOS)
            out.write(struct.pack("<Q", t_us))
            out.write(np.asarray(img, dtype=np.uint8).tobytes())

    dt = (frames[-1][0] - frames[0][0]) / 1e6
    print(f"{args.out}: {len(frames)} frames over {dt:.1f}s "
          f"({(len(frames)-1)/dt:.1f} fps), {len(gyro)} gyro samples, "
          f"fx={fx:.1f} fy={fy:.1f} tilt={tilt:.1f}deg")

if __name__ == "__main__":
    main()
