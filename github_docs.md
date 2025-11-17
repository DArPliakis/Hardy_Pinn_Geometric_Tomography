# Documentation Files

## FILE: docs/installation.md

# Installation Guide

## System Requirements

- **Python**: 3.8 or higher
- **CUDA**: 11.8+ (optional, for GPU acceleration)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 2GB for installation

## Installation Methods

### Method 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/hardy-pinn-geometric-tomography.git
cd hardy-pinn-geometric-tomography

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Method 2: Using conda

```bash
# Clone the repository
git clone https://github.com/yourusername/hardy-pinn-geometric-tomography.git
cd hardy-pinn-geometric-tomography

# Create conda environment
conda env create -f environment.yml
conda activate hardy-pinn

# Install package
pip install -e .
```

### Method 3: Docker (Coming Soon)

```bash
docker pull yourusername/hardy-pinn-tomography:latest
docker run -it hardy-pinn-tomography
```

## Verifying Installation

```python
import torch
from src.solvers.hybrid_solver import HybridPINNFEMSolver

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Quick test
solver = HybridPINNFEMSolver((-1,1,-1,1,-1,1))
print("Installation successful!")
```

## Troubleshooting

### CUDA Issues

If CUDA is not detected:
```bash
# Check CUDA version
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Import Errors

```bash
# Ensure package is installed
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

---

## FILE: docs/quickstart.md

# Quick Start Guide

## 5-Minute Example

```python
from src.solvers.hybrid_solver import HybridPINNFEMSolver
import numpy as np

# 1. Setup domain
domain = (-1, 1, -1, 1, -1, 1)  # (xmin, xmax, ymin, ymax, zmin, zmax)

# 2. Initialize solver
solver = HybridPINNFEMSolver(
    domain_bounds=domain,
    omega=15.0,           # Frequency
    hardy_power=2,        # Hardy weight power
    use_cubic_fem=True,   # Use cubic basis functions
    n_workers=4           # Parallel workers
)

# 3. Setup acquisition geometry
sources = np.array([[-0.9, 0, 0], [-0.9, 0.5, 0.5]])     # Source positions
receivers = np.array([[0.9, 0, 0], [0.9, -0.5, -0.5]])   # Receiver positions

# 4. Synthetic travel time data
observed_times = np.array([1.2, 1.5, 1.3, 1.6])

# 5. Run inversion
history = solver.hybrid_train(
    n_iterations=2,              # Hybrid iterations
    pinn_epochs_per_iter=300,    # PINN epochs
    n_collocation=1500,          # Collocation points
    sources=sources,
    receivers=receivers,
    observed_times=observed_times
)

# 6. Visualize results
solver.visualize_comprehensive()

# 7. Export to VTK
solver.export_results_to_vtk("output/")
```

## Understanding the Parameters

### Domain Setup
- **domain**: 3D box boundaries `(xmin, xmax, ymin, ymax, zmin, zmax)`
- Typically use normalized coordinates: `(-1, 1, -1, 1, -1, 1)`

### Solver Configuration
- **omega**: Frequency parameter (higher = shorter wavelength)
- **hardy_power**: Power for Hardy weight (2 or 3 recommended)
- **use_cubic_fem**: `True` for higher accuracy, `False` for speed
- **n_workers**: Number of CPU cores for parallel ray tracing

### Training Parameters
- **n_iterations**: Number of PINN→FEM→Refine cycles (2-5 typical)
- **pinn_epochs_per_iter**: PINN training epochs (300-1000)
- **n_collocation**: Physics constraint points (1500-3000)

### Data Format
- **sources**: `(n_sources, 3)` array of (x,y,z) coordinates
- **receivers**: `(n_receivers, 3)` array of (x,y,z) coordinates
- **observed_times**: `(n_sources*n_receivers,)` flat array of travel times

## Next Steps

1. **Load Real Data**: See [examples/real_data_example.py](../examples/real_data_example.py)
2. **Tune Parameters**: See [docs/theory.md](theory.md) for guidelines
3. **Visualize in ParaView**: See [docs/paraview_guide.md](paraview_guide.md)
4. **Add Reflections**: See [examples/reflection_example.py](../examples/reflection_example.py)

---

## FILE: docs/theory.md

# Mathematical Theory

## Overview

The Hardy-PINN geometric tomography method combines:
1. **Differential Geometry**: Geodesics in conformally Euclidean metric
2. **Physics-Informed Neural Networks**: Learning with physical constraints
3. **Finite Element Method**: Local accuracy and mesh adaptation
4. **Hardy Inequalities**: A priori estimates for stability

## Conformally Euclidean Metric

The medium is modeled as a conformally Euclidean space:

```
g = (ω/c(x))² δ
```

where:
- `g` is the Riemannian metric tensor
- `ω` is the frequency
- `c(x)` is the spatially-varying wave speed
- `δ` is the Euclidean metric

### Physical Interpretation
- Wave speed `c(x)` determines local propagation velocity
- Conformal factor `ω/c(x)` relates physical to geometric distances
- Geodesics represent ray paths (high-frequency limit)

## Governing Equations

### 1. Wave Equation

```
∆u - (1/c(x)²) ∂²u/∂t² = 0
```

For harmonic solutions `u(x,t) = e^(iωt)v(x)`:

### 2. Helmholtz Equation

```
∆v + (ω²/c(x)²)v = 0
```

### 3. Conformally Weighted Form

With transformation `u = c^(-1/2)v`:

```
∆_g v = -(ω² + Q(x))v
```

where the Q-factor is:

```
Q(x) = (5/4)|∇c|² - (1/2)c∆c
```

### 4. Eikonal Equation

For phase function φ(x):

```
|∇φ|² = 1
```

This is the cornerstone of ray tracing in geometric optics.

## Geodesic Equations

Rays follow geodesics γ(t) satisfying:

```
γ''(t) + Γ^k_ij γ'^i γ'^j = 0
```

For conformal metric, this simplifies to:

```
γ''(t) = -(1/2g)∇g
```

### Jacobi Equation

Geodesic deviation J(t) satisfies:

```
J''(t) - Rm(J, γ')γ' = 0
```

where Rm is the curvature operator. For conformal metrics:

```
Ric = -(2/f)Hess(f) - (1/f²)|∇f|² δ
```

with `f = ω/c`.

## Hardy Inequalities

The Hardy inequality provides a priori estimates:

```
∫ w(x)|v(x)|² dx ≤ C ∫ |∇v(x)|² dx
```

where `w(x) = |x|^p` is the Hardy weight with power p (typically 2 or 3).

### Role in PINNs

Hardy weights regularize the loss function:

```
L_Hardy = ∫ w(x)|∆v + k²v|² dx
```

This ensures:
- **Stability**: Bounds on solution growth
- **Uniqueness**: Prevents spurious modes
- **Convergence**: Guarantees for iterative schemes

## Physics-Informed Loss Function

The total PINN loss combines multiple terms:

```
L_total = λ₁L_eikonal + λ₂L_Helmholtz + λ₃L_conformal + λ₄L_data + λ₅L_FEM
```

### Loss Components

1. **Eikonal Loss**: `L_eikonal = ||∇φ|² - 1|²`
2. **Helmholtz Loss**: `L_Helmholtz = ∫ w(x)|∆v + k²v|² dx`
3. **Conformal Loss**: `L_conformal = ∫ w(x)|∆_g v + (ω²+Q)v|² dx`
4. **Data Loss**: `L_data = |T_computed - T_observed|²`
5. **FEM Consistency**: `L_FEM = |v_PINN - v_FEM|²` at mesh nodes

### Weight Selection

Recommended weights:
- λ₁ = 1.0 (eikonal)
- λ₂ = 1.0 (Helmholtz)
- λ₃ = 1.0 (conformal)
- λ₄ = 10.0 (data fitting - higher priority)
- λ₅ = 2.0 (FEM consistency)

## Hybrid PINN-FEM Algorithm

### Iterative Procedure

```
For iteration k = 1, ..., K:
    1. Train PINN with physics losses + data
    2. Solve FEM on current mesh
    3. Compute residuals on each element
    4. Refine mesh where residual > threshold
    5. Update PINN with FEM consistency loss
End
```

### Convergence Criteria

The method converges when:
1. Data misfit < ε_data (typically 10⁻³)
2. Mesh refinement stabilizes
3. PINN-FEM consistency < ε_FEM (typically 10⁻⁴)

## Adaptive Mesh Refinement

### Residual Indicator

For element e:

```
R_e = ||∆v_h + k²v_h||_e
```

where v_h is the FEM solution.

### Refinement Criterion

Refine element e if:

```
R_e > θ * max(R_e)
```

with threshold θ ∈ [0.1, 0.3].

## Parameter Selection Guidelines

### Frequency ω
- Low (5-10): Smooth variations, faster convergence
- Medium (10-20): Balanced resolution
- High (20+): Fine details, slower convergence

### Hardy Power
- p=2: Standard Hardy inequality
- p=3: Stronger regularization, cubic weight

### Collocation Points
- 1000-1500: Quick testing
- 2000-3000: Production runs
- 5000+: High accuracy requirements

### PINN Epochs
- 200-300: Rapid prototyping
- 500-800: Standard accuracy
- 1000+: High precision

## References

1. Pliakis, D.A. (2025). "Geometric FEM, Ray Tracing, Reflection or Travel Time Tomography and Inversion"
2. Pliakis, D. & Minardi, S. (2009). "Phase front retrieval by means of an iterative shadowgraphic method", J. Opt. Soc. Am. A 26, 99-107

---

## FILE: docs/paraview_guide.md

# ParaView Visualization Guide

## Opening VTK Files

### 1. Launch ParaView
```bash
paraview
```

### 2. Load Files
- **File → Open**
- Navigate to `vtk_output/`
- Select VTK files:
  - `wave_speed_field.vtk` (structured grid)
  - `fem_mesh.vtk` (unstructured grid)
  - `ray_paths.vtk` (polylines)

## Wave Speed Field Visualization

### Volume Rendering
1. Select `wave_speed_field.vtk`
2. Click **Apply**
3. Change representation to **Volume**
4. Adjust **Opacity Transfer Function**
5. Use **Color Map Editor** for color scheme

### Slice View
1. Add filter: **Filters → Alphabetical → Slice**
2. Set slice normal (e.g., Z-axis)
3. Adjust slice position with slider
4. Apply **Contour** filter for isosurfaces

### Isosurfaces
1. Add filter: **Filters → Alphabetical → Contour**
2. Set contour values (e.g., c = 1.0, 1.2, 1.4)
3. Click **Apply**
4. Change color by variable

## FEM Mesh Visualization

### Mesh Structure
1. Select `fem_mesh.vtk`
2. Representation: **Surface With Edges**
3. Color by: `element_size`
4. Shows adaptive refinement

### Solution Field
1. Color by: `fem_solution`
2. Add **Warp By Scalar** for 3D view
3. Scale factor: 0.1-0.5

## Ray Path Visualization

### Trajectories
1. Select `ray_paths.vtk`
2. Representation: **Surface**
3. Color by: `travel_time`
4. Tube filter for thickness:
   - **Filters → Alphabetical → Tube**
   - Radius: 0.01-0.02

### Combined Visualization
1. Load all three files
2. Display simultaneously
3. Adjust opacity for wave speed (0.3)
4. Full opacity for rays and mesh
5. Use **Split Horizontal** for side-by-side

## Advanced Features

### Animations
1. **View → Animation View**
2. Create keyframes
3. Export as MP4 or AVI

### Python Scripting
```python
# ParaView Python script
from paraview.simple import *

# Load data
wavespeed = LegacyVTKReader(FileNames=['wave_speed_field.vtk'])
Show(wavespeed)

# Volume rendering
display = GetDisplayProperties(wavespeed)
display.Representation = 'Volume'

# Render
Render()
```

### Camera Settings
- **Adjust Camera** tool
- Save camera position
- Restore for consistent views

## Export Options

### Images
- **File → Save Screenshot**
- Resolution: 1920×1080 or higher
- Format: PNG (lossless), JPEG (compressed)

### Animations
- **File → Save Animation**
- Frame rate: 30 fps
- Format: MP4, AVI, or image sequence

### 3D Models
- **File → Export Scene**
- Formats: X3D, VRML, POV-Ray

## Tips and Tricks

1. **Performance**: Use **Decimation** filter for large meshes
2. **Transparency**: Adjust opacity curve in Color Map Editor
3. **Comparison**: Use **Calculator** to compute errors
4. **Measurement**: **Ruler** tool for distances
5. **Color Scales**: Save custom color maps

## Troubleshooting

### File Won't Open
- Check VTK format (ASCII vs Binary)
- Verify file integrity
- Check ParaView version (5.0+)

### Slow Performance
- Reduce mesh resolution
- Use **Extract Subset** filter
- Enable GPU rendering in Settings

### Color Issues
- Reset color map to default
- Check data range (rescale to custom range)
- Use logarithmic scale for large ranges

---

## FILE: docs/api_reference.md

# API Reference

## Core Classes

### HybridPINNFEMSolver

Main solver class combining PINN and FEM.

```python
class HybridPINNFEMSolver(domain_bounds, omega=10.0, hardy_power=2, 
                         use_cubic_fem=True, n_workers=None)
```

**Parameters:**
- `domain_bounds`: tuple of 6 floats, (xmin, xmax, ymin, ymax, zmin, zmax)
- `omega`: float, frequency parameter
- `hardy_power`: int, Hardy weight power (2 or 3)
- `use_cubic_fem`: bool, use cubic basis functions
- `n_workers`: int, number of parallel workers (None = auto)

**Methods:**

#### `hybrid_train()`
```python
hybrid_train(n_iterations, pinn_epochs_per_iter, n_collocation,
            sources, receivers, observed_times, 
            reflectors=None, observed_refl_times=None)
```
Main training loop.

**Returns:** list of training history dictionaries

#### `analyze_ray_paths()`
```python
analyze_ray_paths(sources, receivers)
```
Compute geodesic rays and Jacobi fields.

**Returns:** list of ray data dictionaries

#### `export_results_to_vtk()`
```python
export_results_to_vtk(output_dir="vtk_output", ray_data=None)
```
Export results to VTK format.

#### `extract_c_field()`
```python
extract_c_field(resolution=(50, 50, 50))
```
Extract wave speed field on regular grid.

**Returns:** (c_field, x, y, z)

### HardyWeightedPINN

Neural network with Hardy constraints.

```python
class HardyWeightedPINN(input_dim=3, hidden_layers=None, output_dim=1)
```

### GeodesicIntegrator

Geodesic ray tracer.

```python
class GeodesicIntegrator(c_network, omega, device)
```

**Methods:**

#### `integrate_geodesic()`
```python
integrate_geodesic(start_pos, start_dir, n_steps=500, dt=0.005)
```
Integrate geodesic equation.

**Returns:** (trajectory, travel_time)

### JacobiFieldSolver

Solves Jacobi equation.

```python
class JacobiFieldSolver(c_network, omega, device)
```

**Methods:**

#### `solve_jacobi_equation()`
```python
solve_jacobi_equation(geodesic, initial_separation)
```

**Returns:** Jacobi field array

#### `compute_incidence_angle()`
```python
compute_incidence_angle(geodesic, face_normal)
```

**Returns:** float, angle in radians

## Utility Classes

### GPUAccelerator

GPU utilities.

```python
class GPUAccelerator
```

**Static Methods:**
- `get_optimal_device()`: Detect best compute device
- `batch_process(data, model, device, batch_size)`: Process in batches
- `clear_cache()`: Clear GPU cache

### VTKExporter

VTK file export.

```python
class VTKExporter
```

**Static Methods:**
- `export_unstructured_grid()`: Export FEM mesh
- `export_structured_grid()`: Export regular grid
- `export_rays_as_polylines()`: Export ray trajectories

## Examples

### Basic Usage
```python
from src.solvers.hybrid_solver import HybridPINNFEMSolver
import numpy as np

solver = HybridPINNFEMSolver((-1,1,-1,1,-1,1))
sources = np.array([[-0.9, 0, 0]])
receivers = np.array([[0.9, 0, 0]])
times = np.array([1.0])

history = solver.hybrid_train(
    n_iterations=2,
    pinn_epochs_per_iter=300,
    n_collocation=1500,
    sources=sources,
    receivers=receivers,
    observed_times=times
)
```

### Advanced Usage
```python
# With reflection data
reflectors = [(np.array([0, 0, 0]), np.array([0, 0, 1]))]
refl_times = np.array([1.5])

history = solver.hybrid_train(
    n_iterations=3,
    pinn_epochs_per_iter=500,
    n_collocation=2000,
    sources=sources,
    receivers=receivers,
    observed_times=times,
    reflectors=reflectors,
    observed_refl_times=refl_times
)

# Analyze rays
ray_data = solver.analyze_ray_paths(sources[:3], receivers[:3])

# Export
solver.export_results_to_vtk("output/vtk/")
```
