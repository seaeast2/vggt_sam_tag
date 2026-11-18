# P3: KITTI fixed-setting evaluation

This first P3 case follows the predefined evaluation sample order. The CLI accepts a case input run and a separate explicit human approval record; each case is evaluated separately. Input proposals, observations, fitting, and GT evaluation have separate immutable result directories. Future cases require their own actual human polygon approval; do not reuse K02's recorded approval.

- `prepare.py --sample ID --candidate ID --polygons JSON`: RGB-only proposal and frozen solver/settings hashes.
- `observe.py --input RUN --approval JSON`: verifies the approval hashes for both proposed polygons, builds original RGB crops, extracts SIFT scale tracks and LK plane tracks. `lk.py` retains the unchanged extraction block from P1c; `observations.py` is copied unchanged from P1.
- `fit.py`: P2 fit logic with input routing changed; fixed solver snapshot checked by SHA256. Raw depth plus additive residual saved separately.
- `evaluate.py`: reads GT only after fitting, checks immutable fit hashes, generates metrics, image and PLY. Exact rectified camera-Z GT inversion includes calibrated Bz.

Completed runs, 2026-11-17:

```
P3input-kitti-000005-20261117-001
P3obs-kitti-000005-20261117-001
P3-kitti-000005-20261117-001
P3e-kitti-000005-20261117-001
```

CPU execution using cached VGGT-Ω predictions; no new model inference or training. Commands for the recorded fit/evaluation (reruns create new output IDs):

```bash
.venvs/p1/bin/python experiments/planar_residual_depth/P3/fit.py --observations P3obs-kitti-000005-20261117-001
MPLCONFIGDIR=/tmp/p1-mpl .venvs/p1/bin/python experiments/planar_residual_depth/P3/evaluate.py results/planar_residual_depth/P3-kitti-000005-20261117-001
```

See `docs/experiments/planar_residual_depth/P3/P3e-kitti-000005-20261117-001/report.md`. Evaluation IDs 000005 and 000006 are complete; remaining KITTI, Middlebury, ETH3D cases are not claimed complete. ID partition does not certify independent physical scenes or capture sequences.

KITTI 000006 / K03: RGB-only proposal `P3input-kitti-000006-20261117-001` prepared. Both polygons were subsequently approved by the user. Observation run `P3obs-kitti-000006-20261117-001`, fit `P3-kitti-000006-20261117-001`, and evaluation `P3e-kitti-000006-20261117-001` completed. Prior run source snapshots remain immutable.

KITTI 000007 / K04: both polygons explicitly approved. Input `P3input-kitti-000007-20261117-001`, observations `P3obs-kitti-000007-20261117-001`, fit `P3-kitti-000007-20261117-001`, evaluation `P3e-kitti-000007-20261117-001` completed. FIT36/heldout14; frozen solver and immutable fit outputs verified. Evaluation IDs 000005–000007 are now complete; remaining IDs are not claimed complete.

Final continuation: all 29 planned evaluation cases were attempted and P4 is complete. Canonical batch `P3batch-all-evaluation-20261117-001`; analysis `P4-all-evaluation-20261117-001`. 21 cases GT-evaluated, 8 unsupported/no-plane cases retained. See `../P4/README.md`; no further human review is required by the user. Prior completion statements above are historical stage updates.
