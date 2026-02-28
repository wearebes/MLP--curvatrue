# Level-Set Curvature Prediction: Mathematical Formulation

**Version**: 1.0 | **Date**: 2026-02-27

---

## 1. Domain & Discretization

**Computational Domain**:
$$
\Omega = [0,1] \times [0,1]
$$

**Grid Spacing**:
$$
h = \frac{1}{\rho - 1}, \quad \rho \in \{256, 266, 276\}
$$

**Node Coordinates**:
$$
x_i = ih, \quad y_j = jh, \quad i,j = 0, 1, \ldots, \rho-1
$$

**Indexing Convention**:
$$
\phi[i,j] \leftrightarrow \phi(x_i, y_j)
$$

---

## 2. Level-Set Representation

**Interface Definition**:
$$
\Gamma = \{(x,y) \in \Omega : \phi(x,y) = 0\}
$$

**Sign Convention**:
$$
\begin{cases}
\phi > 0 & \text{exterior region} \\
\phi = 0 & \text{interface} \\
\phi < 0 & \text{interior region}
\end{cases}
$$

**Outward Normal**:
$$
\mathbf{n} = \frac{\nabla\phi}{|\nabla\phi|}
$$

**Eikonal Equation** (signed-distance function):
$$
|\nabla\phi| = 1
$$

---

## 3. Geometry Parameters

**Radius Bounds**:
$$
r_{\min} = 1.6h, \quad r_{\max} = 0.5 - 2.0h
$$

**Number of Radii**:
$$
N_r = \left\lfloor \frac{\rho - 8.2}{2} \right\rfloor + 1
$$

**Radius Sampling**:
$$
r_k = r_{\min} + k \cdot \frac{r_{\max} - r_{\min}}{N_r - 1}, \quad k = 0, 1, \ldots, N_r-1
$$

**Center Jitter**:
$$
(c_x, c_y) \sim \mathcal{U}\left(\left[0.5 - \frac{h}{2}, 0.5 + \frac{h}{2}\right]^2\right)
$$

**Center Variations**:
$$
N_v = 5
$$

---

## 4. Level-Set Fields

**Circle SDF** (exact):
$$
\phi_{\text{SDF}}(x,y) = \sqrt{(x-c_x)^2 + (y-c_y)^2} - r
$$

**Circle NonSDF** (requires reinitialization):
$$
\phi_{\text{NonSDF}}(x,y) = (x-c_x)^2 + (y-c_y)^2 - r^2
$$

**Analytical Curvature**:
$$
\kappa = \frac{1}{r}
$$

**Dimensionless Curvature** (target label):
$$
\tilde{\kappa} = h \cdot \kappa = \frac{h}{r}
$$

---

## 5. Curvature Formula

**General Expression**:
$$
\kappa = \nabla \cdot \mathbf{n} = \nabla \cdot \left(\frac{\nabla\phi}{|\nabla\phi|}\right)
$$

**Cartesian Expansion**:
$$
\kappa = \frac{\phi_{xx}\phi_y^2 - 2\phi_{xy}\phi_x\phi_y + \phi_{yy}\phi_x^2}{(\phi_x^2 + \phi_y^2)^{3/2}}
$$

---

## 6. Reinitialization PDE

**Hamilton-Jacobi Equation**:
$$
\frac{\partial\phi}{\partial\tau} + S(\phi_0)\left(|\nabla\phi| - 1\right) = 0, \quad \tau > 0
$$

**Initial Condition**:
$$
\phi(x,y,0) = \phi_0(x,y)
$$

**Smoothed Sign Function**:
$$
S(\phi_0) = \frac{\phi_0}{\sqrt{\phi_0^2 + \varepsilon^2}}, \quad \varepsilon = h
$$

---

## 7. Godunov Upwind Hamiltonian

**Gradient Norm**:
$$
|\nabla\phi|_G = \begin{cases}
\sqrt{\max(D^-_x, -D^+_x, 0)^2 + \max(D^-_y, -D^+_y, 0)^2}, & S(\phi_0) \geq 0 \\[8pt]
\sqrt{\max(-D^-_x, D^+_x, 0)^2 + \max(-D^-_y, D^+_y, 0)^2}, & S(\phi_0) < 0
\end{cases}
$$

where:
- $D^-_x, D^+_x$: Left/right-biased derivatives (WENO5)
- $D^-_y, D^+_y$: Bottom/top-biased derivatives (WENO5)

---

## 8. WENO5 Reconstruction

**Smoothness Indicators**:
$$
\begin{aligned}
\beta_0 &= \frac{13}{12}(v_1 - 2v_2 + v_3)^2 + \frac{1}{4}(v_1 - 4v_2 + 3v_3)^2 \\[5pt]
\beta_1 &= \frac{13}{12}(v_2 - 2v_3 + v_4)^2 + \frac{1}{4}(v_2 - v_4)^2 \\[5pt]
\beta_2 &= \frac{13}{12}(v_3 - 2v_4 + v_5)^2 + \frac{1}{4}(3v_3 - 4v_4 + v_5)^2
\end{aligned}
$$

**Nonlinear Weights** ($\varepsilon_w = 10^{-6}$):
$$
\alpha_k = \frac{d_k}{(\beta_k + \varepsilon_w)^2}, \quad \omega_k = \frac{\alpha_k}{\sum_{j=0}^{2}\alpha_j}
$$

**Linear Weights**:
$$
d_0 = 0.1, \quad d_1 = 0.6, \quad d_2 = 0.3
$$

**Polynomial Approximations**:
$$
\begin{aligned}
p_0 &= \frac{1}{3}v_1 - \frac{7}{6}v_2 + \frac{11}{6}v_3 \\[5pt]
p_1 &= -\frac{1}{6}v_2 + \frac{5}{6}v_3 + \frac{1}{3}v_4 \\[5pt]
p_2 &= \frac{1}{3}v_3 + \frac{5}{6}v_4 - \frac{1}{6}v_5
\end{aligned}
$$

**WENO5 Derivative**:
$$
D = \omega_0 p_0 + \omega_1 p_1 + \omega_2 p_2
$$

**Difference Quotients**:
$$
\delta_{i+1/2} = \frac{\phi_{i+1} - \phi_i}{h}
$$

---

## 9. Time Integration (SSP-RK3)

**Time Step**:
$$
\Delta\tau = C_{\text{CFL}} \cdot h, \quad C_{\text{CFL}} = 0.5
$$

**Right-Hand Side**:
$$
L(\phi) = -S(\phi_0)\left(|\nabla\phi|_G - 1\right)
$$

**Three-Stage Scheme**:
$$
\begin{aligned}
\phi^{(1)} &= \phi^n + \Delta\tau \, L(\phi^n) \\[5pt]
\phi^{(2)} &= \frac{3}{4}\phi^n + \frac{1}{4}\left(\phi^{(1)} + \Delta\tau \, L(\phi^{(1)})\right) \\[5pt]
\phi^{n+1} &= \frac{1}{3}\phi^n + \frac{2}{3}\left(\phi^{(2)} + \Delta\tau \, L(\phi^{(2)})\right)
\end{aligned}
$$

**Reinitialization Steps**:
$$
n \in \{5, 10, 15, 20\}
$$

---

## 10. Interface Band Detection

**Sign Change Conditions**:
$$
\begin{aligned}
\text{x-edge crossing:} & \quad \phi[i,j] \times \phi[i+1,j] \leq 0 \\[5pt]
\text{y-edge crossing:} & \quad \phi[i,j] \times \phi[i,j+1] \leq 0
\end{aligned}
$$

**Boundary Exclusion**:
$$
i,j \in \{1, 2, \ldots, \rho-2\}
$$

---

## 11. Training Feature Extraction

**3×3 Stencil**:
$$
\mathbf{S}_{i,j} = 
\begin{bmatrix}
\phi_{i-1,j+1} & \phi_{i,j+1} & \phi_{i+1,j+1} \\
\phi_{i-1,j} & \phi_{i,j} & \phi_{i+1,j} \\
\phi_{i-1,j-1} & \phi_{i,j-1} & \phi_{i+1,j-1}
\end{bmatrix}
$$

**Feature Vector** (column-major flattening):
$$
\mathbf{x}_{\text{raw}} = [\phi_{i-1,j+1}, \phi_{i,j+1}, \phi_{i+1,j+1}, \phi_{i-1,j}, \phi_{i,j}, \phi_{i+1,j}, \phi_{i-1,j-1}, \phi_{i,j-1}, \phi_{i+1,j-1}]^T
$$

**Scaled Input**:
$$
\mathbf{X} = h \cdot \mathbf{x}_{\text{raw}} \in \mathbb{R}^9
$$

**Target Label**:
$$
Y = h\kappa = \frac{h}{r} \in \mathbb{R}
$$

---

## 12. Data Augmentation

**Sign-Flip Symmetry**:
$$
(\mathbf{X}, Y) \rightarrow \{(\mathbf{X}, Y), (-\mathbf{X}, -Y)\}
$$

**Neural Network Property**:
$$
f_{\text{NN}}(-\mathbf{X}) = -f_{\text{NN}}(\mathbf{X})
$$

---

## 13. Quality Metrics

**Gradient Norm from Stencil**:
$$
\begin{aligned}
\phi_x &\approx \frac{\mathbf{X}[5] - \mathbf{X}[3]}{2h^2} = \frac{\phi_{i+1,j} - \phi_{i-1,j}}{2h} \\[5pt]
\phi_y &\approx \frac{\mathbf{X}[1] - \mathbf{X}[7]}{2h^2} = \frac{\phi_{i,j+1} - \phi_{i,j-1}}{2h}
\end{aligned}
$$

**Eikonal Error** (on interface band $\mathcal{B}$):
$$
\begin{aligned}
\text{MAE} &= \frac{1}{|\mathcal{B}|}\sum_{(i,j) \in \mathcal{B}} \left||\nabla\phi|_{i,j} - 1\right| \\[5pt]
\text{MaxAE} &= \max_{(i,j) \in \mathcal{B}} \left||\nabla\phi|_{i,j} - 1\right| \\[5pt]
\text{Std} &= \sqrt{\frac{1}{|\mathcal{B}|}\sum_{(i,j) \in \mathcal{B}} \left(|\nabla\phi|_{i,j} - \overline{|\nabla\phi|}\right)^2}
\end{aligned}
$$

---

## 14. Dataset Statistics

**Total Blueprints**:
$$
N_{\text{blueprints}} = N_r \times N_v
$$

**Fields per Blueprint**:
$$
N_{\text{fields}} = 1 + 4 = 5 \quad \text{(1 SDF + 4 reinitialized)}
$$

**Samples per Field** (approximate):
$$
N_{\text{samples}} \approx 2 \times 2\pi r \rho \quad \text{(factor of 2 from augmentation)}
$$

**Total Dataset Size**:
$$
N_{\text{total}} \approx 2 \times N_r \times N_v \times N_{\text{fields}} \times \overline{N_{\text{band}}}
$$

**Empirical Results**:
$$
\begin{aligned}
\rho = 256: & \quad N_{\text{total}} = 3{,}137{,}320 \\
\rho = 266: & \quad N_{\text{total}} = 3{,}331{,}060 \\
\rho = 276: & \quad N_{\text{total}} = 3{,}525{,}090
\end{aligned}
$$

---

## 15. Validation Bounds

**Curvature Range**:
$$
\kappa \in \left[\frac{1}{r_{\max}}, \frac{1}{r_{\min}}\right]
$$

**Dimensionless Range**:
$$
h\kappa \in \left[\frac{h}{r_{\max}}, \frac{h}{r_{\min}}\right] = \left[\frac{h}{0.5-2h}, \frac{h}{1.6h}\right] = \left[\frac{h}{0.5-2h}, \frac{1}{1.6}\right]
$$

**For $\rho = 256$** ($h \approx 0.003922$):
$$
h\kappa \in [0.00788, 0.625]
$$

**Symmetry Check**:
$$
\left|\frac{1}{N_{\text{total}}}\sum_{i=1}^{N_{\text{total}}} \mathbf{X}_i\right| < 10^{-6}, \quad \left|\frac{1}{N_{\text{total}}}\sum_{i=1}^{N_{\text{total}}} Y_i\right| < 10^{-6}
$$

**Eikonal Compliance** (SDF samples):
$$
0.9 < \overline{|\nabla\phi|} < 1.1
$$

---

## 16. Key Identities

**Circle Curvature**:
$$
\kappa_{\text{circle}} = \frac{1}{r}
$$

**Zero Level-Set Preservation**:
$$
\phi_0(x,y) = 0 \implies \phi(x,y,\tau) = 0, \quad \forall \tau > 0
$$

**Steady-State Condition**:
$$
\lim_{\tau \to \infty} |\nabla\phi(x,y,\tau)| = 1
$$

**CFL Stability**:
$$
\Delta\tau \leq C_{\text{CFL}} \cdot h
$$

---

## 17. Deterministic Seeding

**Subseed Hash Function**:
$$
\begin{aligned}
s &\leftarrow s_{\text{global}} \\
s &\leftarrow s \oplus 1469598103934665603 \\
s &\leftarrow s \oplus (k+1) \times 1099511628211 \\
s &\leftarrow s \oplus (v+1) \times 14029467366897019727 \\
\text{subseed} &= s \bmod 2^{32}
\end{aligned}
$$

where $k$ = radius index, $v$ = variation index.

---

**End of Mathematical Formulation**

**References**: Jiang & Peng (2000), Shu & Osher (1988), Sussman et al. (1994)
