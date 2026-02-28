# Level-Set Curvature Prediction: Compact Pipeline

**Version**: 1.0 | **Date**: 2026-02-27

---

## Pipeline Overview

```
Blueprint Generation → Field Construction → PDE Reinitialization → Dataset Compilation
    (620 circles)         (SDF + NonSDF)      (WENO5 + RK3)         (3.14M samples)
```

---

## 1. Domain Setup

**Grid**: $\Omega = [0,1]^2$, uniform Cartesian mesh

$$
x_i = ih, \quad y_j = jh, \quad h = \frac{1}{\rho-1}, \quad i,j = 0,\ldots,\rho-1
$$

**Indexing**: `phi[i,j]` $\leftrightarrow$ $\phi(x_i, y_j)$ via `meshgrid(..., indexing='ij')`

**Convention**: $\phi > 0$ (exterior), $\phi < 0$ (interior), $\mathbf{n} = \nabla\phi/|\nabla\phi|$ (outward)

---

## 2. Geometry Generation

**Algorithm 1: Circle Blueprint Generation**

```
Input: ρ, seed, N_v = 5
Output: List of blueprints

1. Compute h = 1/(ρ-1)
2. Set r_min = 1.6h, r_max = 0.5 - 2h
3. Set N_r = ⌊(ρ - 8.2)/2⌋ + 1
4. Generate radii: r_k = linspace(r_min, r_max, N_r), k = 0,...,N_r-1
5. For each k:
   For each v = 0,...,N_v-1:
     a. Compute subseed(k, v) via deterministic hash
     b. Sample (c_x, c_y) ~ U[0.5 - h/2, 0.5 + h/2]²
     c. Store blueprint: {r_k, (c_x, c_y), κ = 1/r_k, label = h/r_k}
6. Return blueprints (total: N_r × N_v)
```

**Key Bounds**:
- Curvature: $\kappa \in [1/r_{\max}, 1/r_{\min}]$
- Dimensionless: $h\kappa \in [h/r_{\max}, h/r_{\min}] \approx [0.008, 0.625]$ for $\rho=256$

---

## 3. Field Construction

**Circle SDF** (exact):
$$
\phi_{\text{SDF}}(x,y) = \sqrt{(x-c_x)^2 + (y-c_y)^2} - r
$$

**Circle NonSDF** (requires reinitialization):
$$
\phi_{\text{NonSDF}}(x,y) = (x-c_x)^2 + (y-c_y)^2 - r^2
$$

Both have **identical** zero level-set: $\Gamma = \{(x,y) : \phi = 0\}$

---

## 4. Reinitialization PDE

**Equation**:
$$
\frac{\partial\phi}{\partial\tau} + S(\phi_0)(|\nabla\phi|_G - 1) = 0, \quad \phi(\tau=0) = \phi_0
$$

**Smoothed Sign**:
$$
S(\phi_0) = \frac{\phi_0}{\sqrt{\phi_0^2 + h^2}}
$$

**Godunov Gradient**:
$$
|\nabla\phi|_G = \begin{cases}
\sqrt{\max(D^-_x, -D^+_x, 0)^2 + \max(D^-_y, -D^+_y, 0)^2}, & S \geq 0 \\
\sqrt{\max(-D^-_x, D^+_x, 0)^2 + \max(-D^-_y, D^+_y, 0)^2}, & S < 0
\end{cases}
$$

---

## 5. WENO5 Spatial Discretization

**Algorithm 2: One-Sided Derivative (HJ-WENO5)**

```
Input: Stencil differences v₁, v₂, v₃, v₄, v₅
Output: High-order approximation D

1. Smoothness indicators:
   β₀ = (13/12)(v₁-2v₂+v₃)² + (1/4)(v₁-4v₂+3v₃)²
   β₁ = (13/12)(v₂-2v₃+v₄)² + (1/4)(v₂-v₄)²
   β₂ = (13/12)(v₃-2v₄+v₅)² + (1/4)(3v₃-4v₄+v₅)²

2. Nonlinear weights (ε = 10⁻⁶):
   α₀ = 0.1/(β₀+ε)², α₁ = 0.6/(β₁+ε)², α₂ = 0.3/(β₂+ε)²
   ω_k = α_k/(α₀+α₁+α₂)

3. Polynomials:
   p₀ = (1/3)v₁ - (7/6)v₂ + (11/6)v₃
   p₁ = -(1/6)v₂ + (5/6)v₃ + (1/3)v₄
   p₂ = (1/3)v₃ + (5/6)v₄ - (1/6)v₅

4. Return D = ω₀p₀ + ω₁p₁ + ω₂p₂
```

**Boundary**: Constant extrapolation (3-layer padding)

---

## 6. Temporal Integration

**Algorithm 3: SSP-RK3 Time Stepping**

```
Input: φ⁰, h, n_steps, CFL = 0.5
Output: φⁿ

1. Set Δτ = CFL·h, S₀ = S(φ⁰)
2. For n = 0 to n_steps-1:
   a. L¹ = -S₀(|∇φⁿ|_G - 1)
      φ⁽¹⁾ = φⁿ + Δτ·L¹
      
   b. L² = -S₀(|∇φ⁽¹⁾|_G - 1)
      φ⁽²⁾ = (3/4)φⁿ + (1/4)(φ⁽¹⁾ + Δτ·L²)
      
   c. L³ = -S₀(|∇φ⁽²⁾|_G - 1)
      φⁿ⁺¹ = (1/3)φⁿ + (2/3)(φ⁽²⁾ + Δτ·L³)
      
3. Return φⁿ
```

**Generated Fields**: {0, 5, 10, 15, 20} steps → 5 fields per blueprint

---

## 7. Interface Band Sampling

**Algorithm 4: Extract Training Nodes**

```
Input: φ (ρ×ρ array)
Output: Lists I, J (node indices)

1. Detect sign changes:
   mask_x = (φ[:-1,:] × φ[1:,:] ≤ 0)  ∧  (boundary exclusion)
   mask_y = (φ[:,:-1] × φ[:,1:] ≤ 0)  ∧  (boundary exclusion)

2. Extract coordinates:
   I_x, J_x = where(mask_x)
   I_y, J_y = where(mask_y)

3. Concatenate (allows duplicates):
   I = [I_x; I_y]
   J = [J_x; J_y]

4. Return I, J
```

**Note**: Nodes with sign changes in **both** directions appear **twice**

---

## 8. Stencil Extraction

**Algorithm 5: 3×3 Stencil Extraction**

```
Input: φ, h, h·κ (label), I, J
Output: X (N×9), Y (N×1)

1. For each (i,j) in zip(I,J):
   a. Extract patch = φ[i-1:i+2, j-1:j+2]
   b. Correct indexing: patch_correct = patch[:, ::-1].T
   c. Flatten: x_raw = patch_correct.flatten()  # 9D vector
   d. Scale: x = h·x_raw
   e. Label: y = h·κ

2. Stack into matrices:
   X_batch = stack(x)  (shape: N×9)
   Y_batch = stack(y)  (shape: N×1)

3. Augmentation:
   X_aug = [X_batch; -X_batch]
   Y_aug = [Y_batch; -Y_batch]

4. Return X_aug, Y_aug
```

**Flattening Order** (column-major after correction):
```
[φ_{i-1,j+1}, φ_{i,j+1}, φ_{i+1,j+1}, 
 φ_{i-1,j},   φ_{i,j},   φ_{i+1,j},
 φ_{i-1,j-1}, φ_{i,j-1}, φ_{i+1,j-1}]
```

---

## 9. Dataset Structure

**HDF5 Schema**:

| Dataset         | Shape    | Type    | Description        |
| --------------- | -------- | ------- | ------------------ |
| `X`             | $(N, 9)$ | float32 | Scaled stencils    |
| `Y`             | $(N, 1)$ | float32 | $h\kappa$ labels   |
| `reinit_steps`  | $(N, 1)$ | int32   | {0, 5, 10, 15, 20} |
| `blueprint_idx` | $(N, 1)$ | int32   | Global ID          |
| `radius_idx`    | $(N, 1)$ | int32   | Radius index       |

**Compression**: GZIP

---

## 10. Quality Metrics

**Eikonal Error** (on interface band):
$$
\text{MAE} = \frac{1}{N_{\text{band}}}\sum_{(i,j)\in\text{band}} \big||\nabla\phi|_{i,j} - 1\big|
$$

**Gradient Approximation**:
$$
\phi_x \approx \frac{X[5] - X[3]}{2h^2}, \quad \phi_y \approx \frac{X[1] - X[7]}{2h^2}
$$

**Expected Convergence**:
- Step 5: MAE ~ $10^{-2}$ to $10^{-3}$
- Step 20: MAE ~ $10^{-4}$ to $10^{-5}$

---

## 11. Validation Protocol

**5-Stage Checks**:

```
✓ Dimension:  X.shape == (N, 9), Y.shape == (N, 1)
✓ Health:     no NaN, no Inf in X, Y
✓ Symmetry:   |mean(X)| < 10⁻⁶, |mean(Y)| < 10⁻⁶
✓ Bounds:     h/r_max ≤ |Y| ≤ h/r_min
✓ Eikonal:    0.9 < mean(|∇φ|) < 1.1 for SDF samples
```

---

## 12. Complexity & Scale

**Per Resolution** ($\rho = 256$):

| Quantity            | Value         |
| ------------------- | ------------- |
| $N_r$ (radii)       | 124           |
| Blueprints          | 620           |
| Fields/blueprint    | 5             |
| Band nodes/field    | ~1010 (avg)   |
| Augmentation factor | 2×            |
| **Total samples**   | **3,137,320** |

**Runtime**: ~30 min/resolution (single CPU)

---

## 13. Implementation Pseudocode

**Main Pipeline**:

```python
for ρ in [256, 266, 276]:
    # Stage 1
    blueprints = CircleGeometryGenerator(ρ, seed=42).generate()
    
    # Stage 2-4
    compiler = HDF5DatasetCompiler(f"train_rho{ρ}.h5")
    for bp in blueprints:
        pack_sdf = build_circle_sdf(bp)
        pack_nonsdf = build_circle_nonsdf(bp)
        
        fields = [pack_sdf]  # step 0
        for n_steps in [5, 10, 15, 20]:
            fields.append(reinitialize(pack_nonsdf, n_steps))
        
        compiler.append(fields)
    compiler.close()
```

---

## 14. Key Constants

```python
# Geometry
r_min = 1.6 * h
r_max = 0.5 - 2.0 * h
N_v = 5

# WENO5
eps_weno = 1e-6
d = [0.1, 0.6, 0.3]

# Time integration
CFL = 0.5
eps_sign_factor = 1.0

# Reinitialization
steps_list = [5, 10, 15, 20]
```

---

## 15. Mathematical Identities

**Curvature** (exact for circles):
$$
\kappa = \frac{1}{r}, \quad h\kappa = \frac{h}{r}
$$

**Eikonal property**:
$$
|\nabla\phi_{\text{SDF}}| = 1 \implies \phi_{xx}\phi_x^2 + 2\phi_{xy}\phi_x\phi_y + \phi_{yy}\phi_y^2 = \phi_x^2 + \phi_y^2
$$

**Sign-flip symmetry**:
$$
f_{\text{NN}}(-\mathbf{X}) = -f_{\text{NN}}(\mathbf{X})
$$

---

## 16. Critical Fixes

| Issue                    | Solution                                     |
| ------------------------ | -------------------------------------------- |
| Python negative indexing | Exclude boundary: $r_{\max} = 0.5 - 2h$      |
| Stencil orientation      | Apply `[:, ::-1].T` after slicing            |
| h5py multiprocessing     | Implement `__getstate__` to drop file handle |
| Label propagation        | Pass `label` dict through all stages         |

---

## References

- **WENO5**: Jiang & Peng (2000), SIAM J. Sci. Comput.
- **SSP-RK3**: Shu & Osher (1988), J. Comput. Phys.
- **Reinitialization**: Sussman et al. (1994), J. Comput. Phys.

---

**End of Compact Documentation**
