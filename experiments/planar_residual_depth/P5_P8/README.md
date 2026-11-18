# P5a–P8: raw distance/confidence normal diagnostics and constrained refinement

User authorized all stages and waived per-region review. Existing frozen P3 masks/observations/cameras/scales are reused. These are **post-hoc exploratory experiments on previously analyzed cases**, not an independent validation benchmark. Confidence means the cached per-pixel VGGT `depth_conf`, not camera pose confidence or calibrated probability.

Canonical runs:

- `results/planar_residual_depth/P5-P8fit-all-cases-20261117-002/`
- `results/planar_residual_depth/P5-P8eval-all-cases-20261117-001/`
- report: `docs/experiments/planar_residual_depth/P5_P8/P5-P8eval-all-cases-20261117-001/report.md`

All P6/P7/P8 rules in `protocol.json` were fixed **before** P5 GT diagnostics. There is no per-case GT winner selection. Main P6 uses the near raw camera-Z third. Confidence-only and near+confidence arms are fixed auxiliary comparisons. P7 uses two independently fitted per-view raw directions and GT-free stability/coverage/confidence weights. P8 constrains the original all-pixel raw normal within 1 degree and optimizes FIT observation loss. All cases succeeded numerically (21 × 6 arms); prior 8 unsupported/no-plane cases remain unrefined. ETH3D five-view fusion is unvalidated.

P5 matches point counts with 20 resamples, and separately matches 4x4-cell counts on common spatial support. If near/mid/far occupy disjoint image areas, the spatial comparison is explicitly unidentifiable, not forced or silently dropped.

Execution, from repository root:

```bash
.venvs/p1/bin/python experiments/planar_residual_depth/P5_P8/run_fit.py
.venvs/p1/bin/python experiments/planar_residual_depth/P5_P8/check_geometry.py
.venvs/p1/bin/python experiments/planar_residual_depth/P5_P8/evaluate.py P5-P8fit-all-cases-20261117-002
MPLCONFIGDIR=/tmp/p1-mpl .venvs/p1/bin/python experiments/planar_residual_depth/P5_P8/report.py P5-P8eval-all-cases-20261117-001
.venvs/p1/bin/python experiments/planar_residual_depth/P5_P8/audit.py P5-P8eval-all-cases-20261117-001
```

Use explicit new IDs for reruns; fit/evaluate create new directories. The first fit run (`...-001`) was preserved and superseded because its administrative routing loader parsed a P4 JSON container containing GT-derived metrics (only routing fields were passed into fitting). Canonical `...-002` obtains routing directly from pre-GT P3 fit manifests and never loads GT metrics during fitting. Original P4 data remain unchanged.

The run uses CPU computation and existing host-GPU raw predictions; no GPU inference, training, SAM, camera optimization or new data selection is performed. Source snapshots are audit copies; restore original source paths for their repository-relative path conventions. Source dependencies from P3 remain frozen and their evidence copies are included.

Outputs include raw/confidence/scale/baseline/masks, selected flattened pixel IDs, raw normal coefficients and bootstrap stability, P5 controls, per-arm residual/corrected depth and planes, FIT/heldout metrics, shared-color-scale RGB/GT/error figures, finite source ROI pointmap arrays with RGB and pixel identity, dataset aggregates and audits. No full-scene or long-running SLAM accuracy claim is made.
