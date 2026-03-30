# Paper Alignment Stage 1

## Changes
- Added parallel explicit-test mode `paper_aligned` without changing the default rebuilt path.
- Added structured paper baselines for the 4 uniform-grid cases from Tables 3, 5, 7, and 8.
- Added a standalone report script to compare `paper_table`, `current_rebuilt`, and `paper_aligned`.
- Made the origin-model report path runnable without TensorFlow and without requiring PyTorch.

## Files
- [generate/test_data.py](/e:/Research/PDE/code1/generate/test_data.py)
- [generate/__main__.py](/e:/Research/PDE/code1/generate/__main__.py)
- [test/paper_reference.py](/e:/Research/PDE/code1/test/paper_reference.py)
- [test/paper_alignment.py](/e:/Research/PDE/code1/test/paper_alignment.py)
- [test/results/paper_alignment_0324.txt](/e:/Research/PDE/code1/test/results/paper_alignment_0324.txt)
- [test/method_alignment.py](/e:/Research/PDE/code1/test/method_alignment.py)
- [test/results/method_alignment_0324.txt](/e:/Research/PDE/code1/test/results/method_alignment_0324.txt)
- [test/explicit_sdf_upper_bound.py](/e:/Research/PDE/code1/test/explicit_sdf_upper_bound.py)
- [test/results/explicit_sdf_upper_bound_0324.txt](/e:/Research/PDE/code1/test/results/explicit_sdf_upper_bound_0324.txt)
- [test/stencil_gap_table.py](/e:/Research/PDE/code1/test/stencil_gap_table.py)
- [test/results/stencil_gap_table_0324.md](/e:/Research/PDE/code1/test/results/stencil_gap_table_0324.md)
- [test/results/stencil_gap_table_0324.csv](/e:/Research/PDE/code1/test/results/stencil_gap_table_0324.csv)
- [test/stencil_repair_ablation.py](/e:/Research/PDE/code1/test/stencil_repair_ablation.py)
- [test/results/stencil_repair_ablation_0324.md](/e:/Research/PDE/code1/test/results/stencil_repair_ablation_0324.md)
- [test/results/stencil_repair_ablation_0324.csv](/e:/Research/PDE/code1/test/results/stencil_repair_ablation_0324.csv)
- [test/reinit_order_ablation.py](/e:/Research/PDE/code1/test/reinit_order_ablation.py)
- [test/results/reinit_order_ablation_0324.md](/e:/Research/PDE/code1/test/results/reinit_order_ablation_0324.md)
- [test/results/reinit_order_ablation_0324.csv](/e:/Research/PDE/code1/test/results/reinit_order_ablation_0324.csv)
- [test/phi_field_quality_table.py](/e:/Research/PDE/code1/test/phi_field_quality_table.py)
- [test/results/phi_field_quality_table_0325_2.md](/e:/Research/PDE/code1/test/results/phi_field_quality_table_0325_2.md)
- [test/results/phi_field_quality_table_0325_2.csv](/e:/Research/PDE/code1/test/results/phi_field_quality_table_0325_2.csv)

## Findings
- `paper_aligned = fixed_phi0_nodes` does not move the results: it matches `current_rebuilt` on all 4 cases and 12 case-steps.
- So the main mismatch with the paper is not explained by "fixed sample nodes vs current sample nodes".
- `target` check: projection target beats radial target on `3/4` cases and `11/12` case-steps, so replacing projection by radial angle does not explain the paper gap.
- Formula and encoding check: no single numerical formula or single stencil order fixes all cases.
- `field_div_normal` gets much closer to the paper numerical row on `smooth_276` and `acute_276`, but is much worse on `smooth_256` and `smooth_266`.
- `origin` input order is only a small residual: `legacy_flat` wins `8/12` steps, but case wins split `2/4` vs `2/4`.
- Scale check: `h`-based input rescaling is not a global fix. It helps `smooth_276` when scaling stencils to the model `h`, and helps `acute_276` slightly in the opposite direction, but effects on `256/266` are negligible.
- Distribution-shift check: rebuilt explicit-test stencils are not wildly out of distribution, but `smooth_276` has the largest drift against `origin/trainStats_276.csv`, and scaling to model `h` reduces that drift consistently.
- The remaining high-priority suspects are input scale or `h / rho_eq` mismatch, numerical-baseline formula and coordinate convention, and smaller residual stencil or axis effects.
- Method check: the local training-data pipeline already matches paper section 3.2 much better than the explicit flower-test pipeline does.
- Train-side spot-check: comparing the local `data02_005/train_rho*.h5` inputs against `origin/trainStats_*.csv` gives max standard-deviation differences of about `1.38e-4` to `1.45e-4`, so the circle-training generator is likely not the main paper-gap source.
- Sampling-rule check: an equation-(13) style "outgoing-edge-only" implementation gives sample counts `320 / 333 / 342 / 406`, so it does not match the paper's `528 / 552 / 564 / 672`; the current two-endpoint deduplicated rule is closer to the paper protocol.
- Reinitializer provenance check: rebuilding the flower tests with a training-style `ij + train reinitializer` pipeline is closer to the paper model on `8/12` steps and `2/4` cases, but only by a small margin. This is a residual improvement, not the main missing piece.
- Exact-SDF upper bound: if the same sample nodes keep the same targets but their 3x3 stencils are replaced by analytic signed-distance values, both the `origin` row and the numerical row get much closer to the paper on `12/12` case-steps.
- This strongly shifts the diagnosis: the main remaining gap is now the irregular explicit-test `phi0 -> reinit -> stencil` chain itself, not target definition, sample-count policy, or a single coordinate flip.
- A quick feature-direction check shows the rebuilt-vs-exact stencil gap is largest at the center and the 4 axis-adjacent entries, with near-symmetric left/right and up/down behavior. That looks more like signed-distance manifold distortion than a simple `x/y` swap bug.
- Stencil-gap table check: in normalized feature space, the dominant rebuilt-vs-exact error group is usually `center`, and `acute_276` is much worse than the smooth cases. The `step=5` smooth cases are the main exception, where diagonal distortion is still comparable or slightly larger.
- Repair ablation check: if only one group is repaired, `cross_exact` is the most effective partial fix on average; `center_exact` helps less and can even make some rows worse; `diag_exact` is consistently the least useful partial repair.
- The strongest takeaway from the ablation is that no single local group repair is enough: `full_exact` wins `12/12` case-steps, while every partial repair remains far from the paper row.
- Reinit-order ablation check: a simple first-order Godunov reinitializer does help a subset of cases, especially `acute_276` and `smooth_276` at step 5, but it is worse overall and does not explain the full gap.
- In the current explicit grid setup, the two WENO5 implementations (`current_test_weno5` and `train_weno5`) are effectively identical in score, so the dominant issue is not just "which WENO5 code path" but the broader reinitialization manifold that the irregular test chain lands on.
- `ij/xy` meshgrid semantics are a real mismatch, but they are not the main error source. An `ij_aligned` rebuild only changes overall RMSE marginally, so this is a residual issue rather than the primary paper-gap cause.
- Field-quality table check on the current `so5_to3` path: the sampled-node eikonal error is already near a plateau by step 5 and barely changes through step 20. The problem is not "still drifting a lot"; it is "stably converged to the wrong manifold."
- The smooth cases all sit around `mean(|grad phi|-1) ≈ 0.24`, while `acute_276` is much worse at about `0.53`, matching its much larger origin/numerical error.
- Against exact-SDF stencils on the same sample nodes, the normalized stencil RMSE is about `0.29` for all smooth cases and about `0.64` for `acute_276`.
- In every case, the dominant normalized feature gap is still the stencil center, with the cross neighbors next and diagonals slightly smaller. This is consistent with a broad signed-distance-shape distortion, not a single axis flip or one-point bug.
- A later evaluation bug was found in [eval_test.py](/e:/Research/PDE/code1/test/eval_test.py): custom `--source` directories were still being rebuilt from the formula `phi0` instead of reading the generated HDF5 payloads. This made early `exact_sdf` reports look falsely bad.
- After fixing that path and re-evaluating the stored [data/exact_sdf](/e:/Research/PDE/code1/data/exact_sdf) payloads, the `origin` row moved to the expected paper-scale errors, e.g. `smooth_256 step 20 -> MSE ≈ 4.43e-07` and `acute_276 step 20 -> MSE ≈ 1.22e-05`, in [exact_sdf_eval_0325_3.txt](/e:/Research/PDE/code1/test/results/exact_sdf_eval_0325_3.txt).

## Current Cause Stack
- Layer 1, mostly ruled out:
  - `origin` model loading and NumPy forward are not the main cause.
  - `fixed_phi0_nodes` is not the main cause.
  - radial-angle `target` is not the main cause.
  - a single stencil-order swap is not the main cause.
- Layer 2, real effects but not a complete explanation:
  - numerical-formula choice matters, but different cases prefer different formulas.
  - `h`-based input rescaling matters, especially in the `276` cases.
  - input-distribution drift exists, but it is moderate rather than catastrophic.
- Layer 3, current main suspects:
  - `276` cases have an extra local scale mismatch between case `h` and `origin/trainStats_276.csv`.
  - numerical-baseline and coordinate conventions are still not fully locked to the paper.
  - smaller stencil and axis residuals are still present, but look secondary.

## Next
- Prioritize the irregular-test generator provenance:
  - current explicit flower tests are a Python rebuild
  - the paper states the irregular section-4 samples came from the authors' original Python/C++ implementations
- Do not spend the next iteration on outgoing-only sampling or simple formula swaps; those are now lower-confidence paths.
- Focus on why the rebuilt flower reinitialization misses the signed-distance manifold:
  - compare rebuilt stencils directly against exact-SDF stencils in normalized feature space
  - isolate whether the remaining gap comes from center-value bias, cross-neighbor bias, or a broader reinitialization-shape distortion

## 276 Focus
- `smooth_276` and `acute_276` do not behave the same way, so they should not be treated as one residual bucket.
- In `smooth_276`, scaling stencils to the model `h` consistently reduces distribution drift against `trainStats_276.csv`.
- In `smooth_276`, `field_div_normal` is much closer than stored `eq3` to the paper numerical row at all 3 steps.
- In `acute_276`, `field_div_normal` wins at steps 5 and 10, but not at step 20.
- In `acute_276`, scaling to model `h` does not help distribution drift; the raw case is already slightly better.
- This points to a mixed cause:
  - one part is `276` scale mismatch, especially for `smooth_276`
  - another part is case-dependent numerical-baseline / coordinate sensitivity
