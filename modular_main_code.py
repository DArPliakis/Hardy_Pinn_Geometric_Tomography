"""
Hardy-PINN Geometric Tomography - Complete Modular Implementation
File: src/solvers/hybrid_solver.py (Main file - split into modules for GitHub)
"""

# =============================================================================
# FILE: src/__init__.py
# =============================================================================
"""Hardy-PINN Geometric Tomography Package"""
__version__ = "1.0.0"
__author__ = "Demetrios A. Pliakis"

from .core.networks import HardyWeightedPINN
from .solvers.hybrid_solver import HybridPINNFEMSolver
from .utils.gpu_acceleration import GPUAccelerator
from .utils.vtk_export import VTKExporter

__all__ = [
    'HardyWeightedPINN',
    'HybridPINNFEMSolver',
    'GPUAccelerator',
    'VTKExporter',
]


# =============================================================================
# FILE: src/core/networks.py
# =============================================================================
"""Neural network architectures for Hardy-weighted PINNs"""
import torch
import torch.nn as nn


class HardyWeightedPINN(nn.Module):
    """
    Physics-Informed Neural Network with Hardy inequality constraints.
    
    Args:
        input_dim: Input dimension (default 3 for x,y,z)
        hidden_layers: List of hidden layer sizes
        output_dim: Output dimension
    """
    
    def __init__(self, input_dim=3, hidden_layers=None, output_dim=1):
        super(HardyWeightedPINN, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 128, 128, 128]
        
        layers = []
        in_features = input_dim
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.Tanh())
            in_features = hidden_size
        
        layers.append(nn.Linear(in_features, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Xavier initialization
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through network"""
        return self.network(x)


# =============================================================================
# FILE: src/core/geometry.py
# =============================================================================
"""Differential geometry components: geodesics and Jacobi fields"""
import numpy as np
import torch


class GeodesicIntegrator:
    """
    Integrates geodesics in conformally Euclidean metric.
    
    The geodesic equation in conformal metric g = f²δ is:
        γ''(t) = -(1/2f²)∇(f²) = -(1/f)∇f
    """
    
    def __init__(self, c_network, omega, device):
        self.c_network = c_network
        self.omega = omega
        self.device = device
    
    def conformal_factor(self, x):
        """Compute g = (ω/c(x))²"""
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            c = self.c_network(x_tensor)
            c = torch.clamp(c, min=0.1)
            g = (self.omega / c)**2
            return g.cpu().numpy()
    
    def compute_gradient_g(self, x, h=1e-5):
        """Compute ∇g using finite differences"""
        g0 = self.conformal_factor(x.reshape(1, -1))
        grad_g = np.zeros(3)
        
        for i in range(3):
            x_plus = x.copy()
            x_plus[i] += h
            g_plus = self.conformal_factor(x_plus.reshape(1, -1))
            grad_g[i] = (g_plus - g0) / h
        
        return grad_g.flatten()
    
    def integrate_geodesic(self, start_pos, start_dir, n_steps=500, dt=0.005):
        """
        Integrate geodesic equation using Verlet-like scheme.
        
        Returns:
            trajectory: (N, 3) array of positions
            travel_time: float, total travel time
        """
        trajectory = [start_pos.copy()]
        position = start_pos.copy()
        velocity = start_dir / np.linalg.norm(start_dir)
        travel_time = 0.0
        
        for step in range(n_steps):
            g = self.conformal_factor(position.reshape(1, -1))[0]
            grad_g = self.compute_gradient_g(position)
            
            # Geodesic acceleration
            if g > 1e-10:
                acceleration = -0.5 * grad_g / g
            else:
                acceleration = np.zeros(3)
            
            # Update
            velocity += acceleration * dt
            velocity = velocity / np.linalg.norm(velocity)
            position += velocity * dt
            travel_time += dt * np.sqrt(g)
            
            trajectory.append(position.copy())
            
            # Boundary check
            if np.any(np.abs(position) > 1.2):
                break
        
        return np.array(trajectory), travel_time


class JacobiFieldSolver:
    """
    Solves Jacobi equation for geodesic deviation.
    
    Jacobi equation: J''(t) - Rm(J, γ')γ' = 0
    where Rm is the curvature operator.
    """
    
    def __init__(self, c_network, omega, device):
        self.c_network = c_network
        self.omega = omega
        self.device = device
    
    def compute_ricci_curvature(self, x, h=1e-5):
        """
        Compute Ricci curvature for conformal metric.
        
        For g = f²δ: Ric = -(2/f)Hess(f) - (1/f²)|∇f|²δ
        """
        with torch.no_grad():
            x_tensor = torch.tensor(x.reshape(1, -1), dtype=torch.float32, 
                                   requires_grad=True).to(self.device)
            c = self.c_network(x_tensor)
            c = torch.clamp(c, min=0.1)
            f = self.omega / c
            
            # Gradient
            grad_f = torch.autograd.grad(f, x_tensor, 
                                        torch.ones_like(f),
                                        create_graph=True)[0]
            
            # Hessian diagonal
            hess_diag = []
            for i in range(3):
                grad_fi = grad_f[0, i]
                hess_ii = torch.autograd.grad(grad_fi, x_tensor,
                                             torch.ones_like(grad_fi),
                                             retain_graph=True)[0][0, i]
                hess_diag.append(hess_ii.item())
            
            f_val = f.item()
            grad_f_norm_sq = torch.sum(grad_f**2).item()
        
        ricci = np.array(hess_diag) * (-2/f_val) - (grad_f_norm_sq/f_val**2)
        return ricci
    
    def solve_jacobi_equation(self, geodesic, initial_separation):
        """
        Solve Jacobi equation along geodesic trajectory.
        
        Args:
            geodesic: (N, 3) array of positions
            initial_separation: (3,) initial separation vector
            
        Returns:
            J: (N, 3) Jacobi field along geodesic
        """
        n_points = len(geodesic)
        J = np.zeros((n_points, 3))
        J_dot = np.zeros((n_points, 3))
        
        J[0] = initial_separation
        J_dot[0] = np.zeros(3)
        
        dt = 0.005
        
        for i in range(1, n_points):
            pos = geodesic[i]
            ricci = self.compute_ricci_curvature(pos)
            
            # Simplified: J'' = -Ric * J
            J_ddot = -ricci * J[i-1]
            
            # Verlet integration
            J[i] = J[i-1] + J_dot[i-1] * dt + 0.5 * J_ddot * dt**2
            J_dot[i] = J_dot[i-1] + J_ddot * dt
        
        return J
    
    def compute_incidence_angle(self, geodesic, face_normal):
        """Compute angle of incidence on a face"""
        direction = geodesic[-1] - geodesic[-2]
        direction = direction / np.linalg.norm(direction)
        cos_angle = np.abs(np.dot(direction, face_normal))
        return np.arccos(np.clip(cos_angle, -1, 1))


# =============================================================================
# FILE: src/solvers/reflection.py
# =============================================================================
"""Reflection tomography handler"""
import numpy as np


class ReflectionHandler:
    """
    Handles reflection tomography data.
    
    Computes reflected ray paths using Fermat's principle.
    """
    
    def __init__(self, geodesic_integrator):
        self.integrator = geodesic_integrator
    
    def compute_reflection_point(self, source, receiver, 
                                 reflector_position, reflector_normal):
        """Find reflection point using mirror reflection (simplified)"""
        d = np.dot(reflector_position - source, reflector_normal)
        reflection_point = source + 2 * d * reflector_normal
        return reflection_point
    
    def compute_reflected_travel_time(self, source, receiver, 
                                     reflector_position, reflector_normal):
        """
        Compute travel time for reflected ray.
        
        Returns:
            total_time: float
            trajectory: (N, 3) combined path
        """
        refl_point = self.compute_reflection_point(
            source, receiver, reflector_position, reflector_normal
        )
        
        # Source to reflection
        dir1 = refl_point - source
        traj1, time1 = self.integrator.integrate_geodesic(source, dir1)
        
        # Reflection to receiver
        dir2 = receiver - refl_point
        traj2, time2 = self.integrator.integrate_geodesic(refl_point, dir2)
        
        return time1 + time2, np.vstack([traj1, traj2])


# =============================================================================
# FILE: src/utils/gpu_acceleration.py
# =============================================================================
"""GPU acceleration utilities"""
import torch


class GPUAccelerator:
    """GPU acceleration and optimization utilities"""
    
    @staticmethod
    def get_optimal_device():
        """Detect and return optimal compute device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        return device
    
    @staticmethod
    def batch_process(data, model, device, batch_size=1000):
        """Process data in batches for memory efficiency"""
        n_samples = len(data)
        results = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = data[i:i+batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
                output = model(batch_tensor)
                results.append(output.cpu().numpy())
        
        return np.vstack(results) if results else np.array([])
    
    @staticmethod
    def clear_cache():
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# FILE: examples/basic_tomography.py
# =============================================================================
"""
Basic tomography example - Quick start guide
"""
import numpy as np
import matplotlib.pyplot as plt
from src.solvers.hybrid_solver import HybridPINNFEMSolver


def run_basic_example():
    """Simple travel time tomography example"""
    
    print("="*70)
    print("Basic Tomography Example")
    print("="*70)
    
    # Setup domain
    domain = (-1, 1, -1, 1, -1, 1)
    
    # Initialize solver
    solver = HybridPINNFEMSolver(
        domain_bounds=domain,
        omega=10.0,
        hardy_power=2,
        use_cubic_fem=False,  # Use linear for speed
        n_workers=2
    )
    
    # Simple acquisition geometry
    n_sources = 5
    n_receivers = 5
    
    sources = np.random.uniform(-0.8, 0.8, (n_sources, 3))
    sources[:, 0] = -0.9  # Fix on one side
    
    receivers = np.random.uniform(-0.8, 0.8, (n_receivers, 3))
    receivers[:, 0] = 0.9  # Fix on opposite side
    
    # Synthetic data
    n_data = n_sources * n_receivers
    observed_times = np.random.uniform(0.8, 1.5, n_data)
    
    print(f"\nAcquisition: {n_sources} sources, {n_receivers} receivers")
    print(f"Data points: {n_data}")
    
    # Run inversion (quick)
    history = solver.hybrid_train(
        n_iterations=1,
        pinn_epochs_per_iter=200,
        n_collocation=1000,
        sources=sources,
        receivers=receivers,
        observed_times=observed_times
    )
    
    # Visualize
    fig = solver.visualize_comprehensive()
    plt.savefig('basic_example_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nExample complete! Result saved to basic_example_result.png")


if __name__ == "__main__":
    run_basic_example()


# =============================================================================
# FILE: scripts/train.py
# =============================================================================
"""
Main training script with command-line interface
"""
import argparse
import numpy as np
from pathlib import Path
from src.solvers.hybrid_solver import HybridPINNFEMSolver


def main():
    parser = argparse.ArgumentParser(
        description='Hardy-PINN Geometric Tomography Training'
    )
    
    parser.add_argument('--omega', type=float, default=15.0,
                       help='Frequency parameter')
    parser.add_argument('--hardy_power', type=int, default=2,
                       help='Hardy weight power')
    parser.add_argument('--n_iterations', type=int, default=3,
                       help='Number of hybrid iterations')
    parser.add_argument('--pinn_epochs', type=int, default=500,
                       help='PINN epochs per iteration')
    parser.add_argument('--n_collocation', type=int, default=2000,
                       help='Number of collocation points')
    parser.add_argument('--n_sources', type=int, default=12,
                       help='Number of sources')
    parser.add_argument('--n_receivers', type=int, default=12,
                       help='Number of receivers')
    parser.add_argument('--use_cubic', action='store_true',
                       help='Use cubic FEM basis functions')
    parser.add_argument('--n_workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--export_vtk', action='store_true',
                       help='Export VTK files for ParaView')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("Hardy-PINN Geometric Tomography - Training")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Omega: {args.omega}")
    print(f"  Hardy power: {args.hardy_power}")
    print(f"  Iterations: {args.n_iterations}")
    print(f"  PINN epochs: {args.pinn_epochs}")
    print(f"  Collocation points: {args.n_collocation}")
    print(f"  Cubic FEM: {args.use_cubic}")
    print(f"  Workers: {args.n_workers}")
    print(f"  Output: {args.output_dir}")
    
    # Initialize solver
    domain = (-1, 1, -1, 1, -1, 1)
    solver = HybridPINNFEMSolver(
        domain_bounds=domain,
        omega=args.omega,
        hardy_power=args.hardy_power,
        use_cubic_fem=args.use_cubic,
        n_workers=args.n_workers
    )
    
    # Setup acquisition
    sources = np.random.uniform(-0.8, 0.8, (args.n_sources, 3))
    sources[:, 0] = -0.9
    
    receivers = np.random.uniform(-0.8, 0.8, (args.n_receivers, 3))
    receivers[:, 0] = 0.9
    
    # Synthetic data (replace with real data loading)
    n_data = args.n_sources * args.n_receivers
    observed_times = np.random.uniform(0.8, 1.8, n_data)
    
    # Train
    history = solver.hybrid_train(
        n_iterations=args.n_iterations,
        pinn_epochs_per_iter=args.pinn_epochs,
        n_collocation=args.n_collocation,
        sources=sources,
        receivers=receivers,
        observed_times=observed_times
    )
    
    # Save results
    fig = solver.visualize_comprehensive()
    plt.savefig(output_path / 'inversion_result.png', dpi=150, bbox_inches='tight')
    
    if args.export_vtk:
        solver.export_results_to_vtk(str(output_path / 'vtk'))
    
    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"Results saved to {args.output_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()


# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================
"""
To use this modular code:

1. Split this file into the individual files as indicated by comments
2. Create the directory structure as shown in the repository layout
3. Place each section in its corresponding file
4. Install the package: pip install -e .
5. Run examples or use the training script

Example commands:

# Basic example
python examples/basic_tomography.py

# Full training with options
python scripts/train.py --omega 15 --n_iterations 3 --use_cubic --export_vtk

# From Python
from src.solvers.hybrid_solver import HybridPINNFEMSolver
solver = HybridPINNFEMSolver(domain_bounds=(-1,1,-1,1,-1,1))
"""
