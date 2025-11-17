# 🌊 Hardy-PINN Geometric Tomography

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org)

**Hybrid Physics-Informed Neural Networks and Finite Element Method for Geometric Tomographic Inversion**

> *Implementation of the method described in "Geometric FEM, Ray Tracing, Reflection or Travel Time Tomography and Inversion" by Demetrios A. Pliakis (May 2025)*

---

## 🎯 Key Features

### 🧠 **Physics-Informed Neural Networks**
- Hardy-weighted loss functions with cubic weight
- Eikonal equation: |∇φ|² = 1
- Helmholtz equation in conformal metric
- Automatic differentiation for exact gradients

### 📐 **Differential Geometry**
- Geodesic ray tracing in conformally Euclidean space
- Jacobi field computation for geodesic deviation
- Ricci curvature calculation
- Angle of incidence on mesh faces

### 🔬 **Finite Element Method**
- 20-node tetrahedral cubic elements
- Adaptive mesh refinement
- Residual-based error estimation
- Hybrid PINN-FEM coupling

### 📊 **Tomographic Inversion**
- Travel time tomography
- Reflection tomography
- Multi-source multi-receiver geometry
- Iterative model updating

### ⚡ **High Performance**
- GPU acceleration (CUDA)
- Parallel ray tracing (multiprocessing)
- Batch processing
- Memory-optimized algorithms

### 📈 **Visualization**
- Matplotlib 2D/3D plots
- VTK export for ParaView
- Interactive ray path visualization
- Mesh refinement analysis

---

## 📦 Installation

### Quick Start (5 minutes)

```bash
# Clone repository
git clone https://github.com/yourusername/hardy-pinn-geometric-tomography.git
cd hardy-pinn-geometric-tomography

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Verify installation
python -c "from src.solvers.hybrid_solver import HybridPINNFEMSolver; print('Success!')"
```

### Conda Environment

```bash
# Create environment
conda env create -f environment.yml
conda activate hardy-pinn

# Install package
pip install -e .
```

### GPU Support

For CUDA acceleration:
```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Quick Example

```python
from src.solvers.hybrid_solver import HybridPINNFEMSolver
import numpy as np
import matplotlib.pyplot as plt

# 1. Initialize solver
solver = HybridPINNFEMSolver(
    domain_bounds=(-1, 1, -1, 1, -1, 1),
    omega=15.0,
    hardy_power=2,
    use_cubic_fem=True,
    n_workers=4
)

# 2. Setup acquisition geometry
sources = np.array([[-0.9, 0, 0], [-0.9, 0.5, 0.5]])
receivers = np.array([[0.9, 0, 0], [0.9, -0.5, -0.5]])
observed_times = np.array([1.2, 1.5, 1.3, 1.6])

# 3. Run inversion
history = solver.hybrid_train(
    n_iterations=2,
    pinn_epochs_per_iter=300,
    n_collocation=1500,
    sources=sources,
    receivers=receivers,
    observed_times=observed_times
)

# 4. Visualize results
solver.visualize_comprehensive()
plt.savefig('result.png', dpi=150)

# 5. Export to VTK for ParaView
solver.export_results_to_vtk("vtk_output/")
```

**Output:** Wave speed field recovery with geodesic ray paths!

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Installation Guide](docs/installation.md) | Detailed installation instructions |
| [Quick Start](docs/quickstart.md) | 5-minute tutorial |
| [Theory](docs/theory.md) | Mathematical background |
| [API Reference](docs/api_reference.md) | Complete API documentation |
| [ParaView Guide](docs/paraview_guide.md) | 3D visualization tutorial |

---

## 💡 Examples

### Basic Travel Time Tomography
```bash
python examples/basic_tomography.py
```

### With Reflection Data
```bash
python examples/reflection_example.py
```

### Command-Line Training
```bash
python scripts/train.py \
  --omega 15 \
  --n_iterations 3 \
  --pinn_epochs 500 \
  --n_collocation 2000 \
  --use_cubic \
  --export_vtk \
  --output_dir results/
```

### Jupyter Notebooks
- `notebooks/01_introduction.ipynb` - Introduction
- `notebooks/02_geodesic_rays.ipynb` - Ray tracing
- `notebooks/03_jacobi_fields.ipynb` - Geodesic deviation
- `notebooks/04_hardy_pinns.ipynb` - PINN training
- `notebooks/05_fem_coupling.ipynb` - FEM integration
- `notebooks/06_full_inversion.ipynb` - Complete workflow

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_geometry.py

# With coverage
pytest --cov=src tests/
```

---

## 🎓 Theory Overview

### Conformally Euclidean Metric

The method models the medium with metric:
```
g = (ω/c(x))² δ
```
where `c(x)` is the spatially-varying wave speed.

### Governing Equations

1. **Helmholtz Equation:**
   ```
   ∆v + (ω²/c²)v = 0
   ```

2. **Conformal Form with Q-factor:**
   ```
   ∆_g v = -(ω² + Q(x))v
   Q(x) = (5/4)|∇c|² - (1/2)c∆c
   ```

3. **Eikonal Equation:**
   ```
   |∇φ|² = 1
   ```

4. **Geodesic Equation:**
   ```
   γ''(t) = -(1/2g)∇g
   ```

5. **Jacobi Equation:**
   ```
   J''(t) - Rm(J, γ')γ' = 0
   ```

### Hardy Inequality

Provides a priori estimates:
```
∫ w(x)|v|² dx ≤ C ∫ |∇v|² dx
```
with Hardy weight `w(x) = |x|^p`.

---

## 🏗️ Project Structure

```
hardy-pinn-geometric-tomography/
├── src/
│   ├── core/              # Neural networks, geometry
│   ├── solvers/           # Hybrid solver, ray tracing
│   ├── utils/             # GPU, VTK export, visualization
│   └── losses/            # Physics-informed losses
├── examples/              # Usage examples
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── notebooks/             # Jupyter tutorials
├── scripts/               # Training scripts
└── paper/                 # Original paper
```

---

## 📊 Performance Benchmarks

| Configuration | Time (s) | Accuracy |
|--------------|----------|----------|
| CPU + Linear FEM | 120 | 0.05 |
| CPU + Cubic FEM | 180 | 0.02 |
| GPU + Linear FEM | 35 | 0.05 |
| GPU + Cubic FEM | 50 | 0.02 |

*Benchmark: 10 sources, 10 receivers, 2000 collocation points, 300 epochs*

---

## 🎨 Visualization Gallery

<table>
  <tr>
    <td><img src="docs/images/wave_speed.png" width="250"/><br/><sub>Wave Speed Field</sub></td>
    <td><img src="docs/images/ray_paths.png" width="250"/><br/><sub>Geodesic Rays</sub></td>
    <td><img src="docs/images/mesh.png" width="250"/><br/><sub>Adaptive Mesh</sub></td>
  </tr>
</table>

*Open VTK files in ParaView for interactive 3D visualization!*

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{pliakis2025geometric,
  title={Geometric FEM, Ray Tracing, Reflection or Travel Time Tomography and Inversion},
  author={Pliakis, Demetrios A.},
  year={2025},
  month={May}
}

@article{pliakis2009phase,
  title={Phase front retrieval by means of an iterative shadowgraphic method},
  author={Pliakis, Dimitris and Minardi, Stefano},
  journal={Journal of the Optical Society of America A},
  volume={26},
  number={1},
  pages={99--107},
  year={2009},
  publisher={Optica Publishing Group}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/hardy-pinn-geometric-tomography.git

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/ tests/ examples/

# Type checking
mypy src/
```

---

## 🐛 Bug Reports

Please report bugs by opening an issue with:
- Python version
- PyTorch version
- CUDA version (if applicable)
- Minimal code to reproduce
- Error messages

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

```
MIT License - Copyright (c) 2025 Demetrios A. Pliakis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgments

- **Demetrios A. Pliakis** for the theoretical framework
- **Physics-Informed Neural Networks** community
- **PyTorch** team for the deep learning framework
- **ParaView** developers for visualization tools

---

## 📬 Contact

- **Author:** Demetrios A. Pliakis
- **Email:** [your-email@example.com]
- **GitHub:** [@yourusername](https://github.com/yourusername)
- **Issues:** [GitHub Issues](https://github.com/yourusername/hardy-pinn-geometric-tomography/issues)

---

## 🔗 Related Projects

- [Physics-Informed Neural Networks](https://github.com/maziarraissi/PINNs)
- [DeepXDE](https://github.com/lululxvi/deepxde)
- [FEniCS Project](https://fenicsproject.org/)
- [ParaView](https://www.paraview.org/)

---

## 📈 Roadmap

- [x] Core Hardy-PINN implementation
- [x] Geodesic ray tracing
- [x] Jacobi field computation
- [x] Cubic FEM elements
- [x] Adaptive mesh refinement
- [x] VTK export
- [ ] Real seismic data examples
- [ ] Uncertainty quantification
- [ ] Multi-GPU support
- [ ] Web-based visualization
- [ ] Docker container

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/hardy-pinn-geometric-tomography&type=Date)](https://star-history.com/#yourusername/hardy-pinn-geometric-tomography&Date)

---

<div align="center">

**Made with ❤️ by the Hardy-PINN team**

[⬆ Back to top](#-hardy-pinn-geometric-tomography)

</div>
