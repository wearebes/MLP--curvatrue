# Method Report: `dynamic_phi` with `cfl=0.5`, `eps_sign_factor=2.0`

## Scope
This report documents the current preferred method when:

- `cfl` must remain fixed at `0.5`
- the outer irregular-flower testing pipeline should remain unchanged
- only reinitialization details are allowed to change

The method reported here is:

- `phi0 mode = formula_phi0`
- `sign_mode = dynamic_phi`
- `cfl = 0.5`
- `eps_sign_factor = 2.0`
- `time_order = 3`
- `space_order = 5`
- `sampling_rule = current_interface_nodes`
- `target_rule = analytic_projection_current_nodes`

This is the strongest configuration found so far under the constraint that `cfl` stays at `0.5`.

## Pipeline
The full testing pipeline is:

1. Build the algebraic flower initial level-set field

   `phi0(x,y) = sqrt(x^2 + y^2) - a cos(p theta) - b`

2. Reinitialize `phi0` using the level-set reinitialization PDE with:

   - `dynamic_phi`
   - `cfl=0.5`
   - `eps_sign_factor=2.0`
   - TVD-RK3 in time
   - WENO5 in space

3. Extract `current_interface_nodes` from the reinitialized field

4. For each sampled node, extract a `3x3` stencil encoded in `training_order`

5. For each sampled node, compute the target using analytic projection:

   - project the node onto the flower interface with Newton projection
   - evaluate the analytic curvature at the projected angle
   - store `hkappa_target = h * kappa_exact`

6. Evaluate three predictors on the same samples:

   - `Numerical`: equation (3) / expanded finite-difference curvature
   - `Origin`: the bundled reference model
   - `Neural`: `model_data02_002`

7. Report:

   - `MAE_hk`
   - `MaxAE_hk`
   - `MSE_hk`

## Why This Method
Under fixed `cfl=0.5`, the key finding was:

- the original `frozen_phi0 + eps_sign_factor=1.0` reinitializer is too inaccurate
- switching only to `dynamic_phi` helps, but is still far from paper
- increasing `eps_sign_factor` while keeping `cfl=0.5` closes most of the remaining gap

For this reason, the preferred fixed-`cfl` method is:

`formula_phi0 -> dynamic_phi reinit(cfl=0.5, eps_sign_factor=2.0, RK3, WENO5) -> current_interface_nodes -> 3x3 stencil -> projection target -> numerical/origin/neural`

## Dataset Used
Generated dataset:

- `data/paper_formula_cfl05_eps20_dynsign`

All sample counts match the paper:

- `smooth_256`: `528`
- `smooth_266`: `552`
- `smooth_276`: `564`
- `acute_276`: `672`

## Result Tables
Raw evaluation snapshot:

- `test/results/20260327_H08_method_report_cfl05_eps20_eval.txt`

### Neural
```text
[test] Neural | model=model_data02_002 | curvature=model_prediction | nodes=current_interface_nodes | source=paper_formula_cfl05_eps20_dynsign

         group_id  rho_model  step_or_iter  N_samples       MAE_hk     MaxAE_hk       MSE_hk
smooth/smooth_256        256             5        528 2.114708e-03 1.946637e-02 9.846444e-06
smooth/smooth_256        256            10        528 1.057069e-03 5.395644e-03 1.979274e-06
smooth/smooth_256        256            20        528 9.542494e-04 4.974988e-03 1.554504e-06
smooth/smooth_266        266             5        552 1.812341e-03 1.249766e-02 6.001427e-06
smooth/smooth_266        266            10        552 1.023693e-03 5.592954e-03 1.924138e-06
smooth/smooth_266        266            20        552 9.451274e-04 5.081722e-03 1.596934e-06
smooth/smooth_276        276             5        564 2.454514e-03 1.581745e-02 1.253388e-05
smooth/smooth_276        276            10        564 1.109531e-03 7.480319e-03 2.325846e-06
smooth/smooth_276        276            20        564 9.629054e-04 6.863113e-03 1.713370e-06
  acute/acute_276        276             5        672 5.731719e-03 6.646543e-02 7.671422e-05
  acute/acute_276        276            10        672 2.865863e-03 5.691286e-02 2.985668e-05
  acute/acute_276        276            20        672 2.631278e-03 5.300706e-02 2.874231e-05
```

### Numerical
```text
[test] Numerical | model=baseline_numerical | curvature=expanded_formula | nodes=current_interface_nodes | source=paper_formula_cfl05_eps20_dynsign

         group_id  rho_model  step_or_iter  N_samples       MAE_hk     MaxAE_hk       MSE_hk
smooth/smooth_256        256             5        528 1.493764e-03 8.385750e-03 3.934469e-06
smooth/smooth_256        256            10        528 1.464285e-03 1.293389e-02 4.687282e-06
smooth/smooth_256        256            20        528 1.418718e-03 1.472805e-02 4.946166e-06
smooth/smooth_266        266             5        552 1.398437e-03 6.192293e-03 3.373041e-06
smooth/smooth_266        266            10        552 1.392637e-03 9.598911e-03 4.070944e-06
smooth/smooth_266        266            20        552 1.330652e-03 1.209511e-02 4.326671e-06
smooth/smooth_276        276             5        564 1.351475e-03 7.240336e-03 3.461005e-06
smooth/smooth_276        276            10        564 1.354251e-03 1.103839e-02 4.218635e-06
smooth/smooth_276        276            20        564 1.309090e-03 1.238965e-02 4.423052e-06
  acute/acute_276        276             5        672 3.662445e-03 6.604158e-02 5.761354e-05
  acute/acute_276        276            10        672 3.458457e-03 8.523055e-02 6.549082e-05
  acute/acute_276        276            20        672 3.325215e-03 9.166238e-02 6.884664e-05
```

### Origin
```text
[test] Origin | model=origin | curvature=model_prediction | nodes=current_interface_nodes | source=paper_formula_cfl05_eps20_dynsign

         group_id  rho_model  step_or_iter  N_samples       MAE_hk     MaxAE_hk       MSE_hk
smooth/smooth_256        256             5        528 1.796834e-03 8.752025e-03 6.056347e-06
smooth/smooth_256        256            10        528 8.799425e-04 4.565212e-03 1.430765e-06
smooth/smooth_256        256            20        528 7.903758e-04 5.133840e-03 1.148823e-06
smooth/smooth_266        266             5        552 1.610112e-03 9.949934e-03 4.835201e-06
smooth/smooth_266        266            10        552 8.961656e-04 7.247397e-03 1.619881e-06
smooth/smooth_266        266            20        552 8.306593e-04 7.585028e-03 1.340135e-06
smooth/smooth_276        276             5        564 1.854380e-03 9.964302e-03 5.716208e-06
smooth/smooth_276        276            10        564 1.064377e-03 6.244197e-03 1.902919e-06
smooth/smooth_276        276            20        564 8.992267e-04 4.980922e-03 1.439912e-06
  acute/acute_276        276             5        672 3.910179e-03 6.208190e-02 3.914207e-05
  acute/acute_276        276            10        672 2.746620e-03 5.139002e-02 2.596395e-05
  acute/acute_276        276            20        672 2.573224e-03 4.679858e-02 2.548924e-05
```

## Paper-Relative Summary
Average MSE ratios against the paper tables:

- `Numerical / paper_numerical = 0.9516`
- `Origin / paper_model = 2.0934`
- `model_data02_002 / paper_model = 2.9208`

Range across the 12 case-steps:

- `Numerical`: min `0.3445`, max `1.3767`
- `Origin`: min `1.4401`, max `2.3584`
- `model_data02_002`: min `2.3087`, max `5.1713`

## Interpretation
This method is the strongest option found so far under fixed `cfl=0.5`.

What it achieves:

- sample counts still match the paper exactly
- numerical MSE is already very close to the paper numerical row on average
- the outer testing pipeline does not need to change
- only the reinitializer internals are adjusted

What it does not achieve:

- `Origin` is still about `2.09x` the paper model MSE on average
- `model_data02_002` is still about `2.92x` the paper model MSE on average
- so this method is a strong fixed-`cfl` compromise, but not the near-paper best overall configuration

## Best Use
Use this method when:

- you want to keep `cfl=0.5`
- you want a method-level reinit improvement
- you do not want to move all the way to the more aggressive near-paper configuration

Do not use this report to claim full paper alignment. For that, the stronger configuration remains:

- `dynamic_phi`
- `cfl = 0.95`
- `eps_sign_factor = 2.5`
- `RK3`
- `WENO5`

## Reproduction
Dataset generation in the current repo was done with a one-off script that produced:

- `data/paper_formula_cfl05_eps20_dynsign`

Evaluation command:

```bash
python -m test numerical origin neural --data-split test --run-id model_data02_002 --source paper_formula_cfl05_eps20_dynsign
```

WSL environment used for the neural model run:

- conda env: `jq`
- Python: `/home/yjc/miniconda3/envs/jq/bin/python`
- Torch: `2.10.0+cu130`


formula_phi0
-> dynamic_phi reinitialization (cfl=0.95, eps_sign_factor=2.5, RK3, WENO5)
-> current_interface_nodes
-> 3x3 stencil (training_order)
-> projection-based analytic hkappa_target
-> numerical / origin / model_data02_002
-> MAE / MaxAE / MSE
