# P9a–P9d: direct RGB planar alignment

Canonical runs:

- `results/planar_residual_depth/P9fit-all-cases-20261117-001/`
- `results/planar_residual_depth/P9eval-all-cases-20261117-001/`
- Report: `docs/experiments/planar_residual_depth/P9/P9eval-all-cases-20261117-001/report.md`

All settings were fixed in `protocol.json` before fitting and before GT diagnosis. Existing 21 cases were attempted with immutable original masks/cameras/scales. This is an exploratory extension of already analyzed cases, not independent validation. Original 8 unsupported cases remain unsupported. P9b/c photometric refinement is supported in 13 KITTI and 3 Middlebury cases. Five other existing cases retain B2 because guarded FIT/heldout spatial blocks are insufficient; thresholds are not relaxed.

`photo.py` implements analytic planar projection, exact floating-point bilinear RGB derivatives, RGB Huber loss, gain/bias, positive depth and +/-3 processed-pixel corner warp trust constraints. P9b fixes the original normal and optimizes distance. P9c optimizes all three plane parameters. Only photo FIT pixels are used for geometry and brightness selection. Original B2/B3 remain controls; `F9C_MATCH_TRUST` applies the original feature objective with the same trust constraints.

FIT/heldout are distinct canonical source32px blocks, with8px guards. Source/target mask erosion and common B2/B3 initial footprint freeze the support before fitting; no pixels are dropped according to current residual. Photometric heldout means unused by P9 refinement, not completely unseen by the original RGB model/LK initialization. Guarded split support is explicitly audited.

Execution from repository root (pass newly returned run IDs on reruns):

```bash
.venvs/p1/bin/python experiments/planar_residual_depth/P9/check_photo.py
.venvs/p1/bin/python experiments/planar_residual_depth/P9/check_fixed.py
.venvs/p1/bin/python experiments/planar_residual_depth/P9/run_fit.py
.venvs/p1/bin/python experiments/planar_residual_depth/P9/pre_gt_audit.py P9fit-all-cases-20261117-001
.venvs/p1/bin/python experiments/planar_residual_depth/P9/evaluate.py P9fit-all-cases-20261117-001
MPLCONFIGDIR=/tmp/p1-mpl .venvs/p1/bin/python experiments/planar_residual_depth/P9/report.py P9eval-all-cases-20261117-001
.venvs/p1/bin/python experiments/planar_residual_depth/P9/audit.py P9eval-all-cases-20261117-001
```

The separate pre-GT audit completes the scalar distance-photo Jacobian observability check; all supported cases passed, so no post-GT geometry/support selection occurs. Evaluation only then reads GT and additionally renders the GT reference plane as an **oracle diagnostic**, not an initializer or candidate selector.

P9d reports appearance mismatch, texture, affine brightness bounds and warp bounds. No real occlusion GT is available, so appearance residual is not claimed as occlusion detection accuracy. Synthetic constant-texture and modified-patch checks validate the diagnostic code only.

The mean comparison is per-dataset and per-case, and reports both paired supported cases and all retained cases with B2 fallback. ID bootstrap intervals are descriptive because cases are previously seen and may share physical scenes. A normal improvement must exceed1e-6 degrees to avoid counting scalar-rescaling floating-point roundoff as direction correction. The initial report before this reporting-only correction is archived; fit outputs remain unchanged.

No new GPU inference, network training, segmentation, camera refinement, or long-duration SLAM occurs; CPU computation reuses existing host-GPU prediction caches. Outputs preserve raw/confidence/scale, residual/corrected depth, affine parameters, plane coefficients, support pixel coordinates, geometry/photometric diagnostics, source RGB/warps/checkerboards/error maps, finite ROI pointmaps with RGB/pixel identity, and GT provenance/hash audits.
