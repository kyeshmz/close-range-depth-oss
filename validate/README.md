# Side-by-side validation (Jetson = reference rig)

## 1. On the Jetson (has camera + vendor `rs_depth`)

```bash
python3 validate/jetson_capture.py --out runs/d405_close --frames 200
```

Point the D400 at close-range scenes (120–500 mm): textured planes,
edges, the actual manipulation/inspection targets. Saves one `.npz` per
frame: raw IR pair + HW depth + vendor output + calibration + timings.

## 2. Copy back and score (this box or anywhere)

```bash
scp -r jetson:runs/d405_close runs/
python3 validate/compare.py --data runs/d405_close --out runs/compare
```

Writes `metrics.csv` (per frame: band pixel counts, rescued counts,
vendor-vs-ours `median_rel_err` / `agree10`, vendor timing) and
`frame_*.png` triples: **HW | vendor | ours**, identical color scale
capped at the close-range threshold.

## What good looks like

* `rescued_ours` ≈ `rescued_vendor` (we recover the same blind pixels)
* `agree10` high / `median_rel_err` low where both are valid
* ours timing vs `t_vendor_ms` column
