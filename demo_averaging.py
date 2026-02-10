"""
Demo script for multi-simulation averaging functionality.

This script demonstrates how to run multiple simulations, collect results,
and visualize the average positions and displacement arrows.
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_gravity_simulator import (
    Network1D,
    Network2DSquare,
    Network2DTriangular,
    run_multiple_simulations,
    visualize_averaged_results
)


def demo_1d_averaging():
    """Demonstrate multi-simulation averaging for 1D network."""
    print("\n" + "="*60)
    print("DEMO: Multi-Simulation Averaging - 1D Chain")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Network parameters
    network_params = {
        'n_masses': 11,
        'mass': 1.0,
        'spring_constant': 10.0,
        'damping': 0.5,
        'dt': 0.01,
        'temperature': 2.0
    }
    
    # Run multiple simulations
    results = run_multiple_simulations(
        network_class=Network1D,
        network_params=network_params,
        simulation_steps=1000,
        n_simulations=10,
        show_progress=True
    )
    
    # Visualize averaged results
    print("\nGenerating averaged visualization...")
    fig = visualize_averaged_results(
        results=results,
        network_params=network_params,
        title="1D Mass-Spring Chain - Averaged Over 10 Simulations"
    )
    
    filename = 'averaged_1d_chain.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {filename}")
    plt.close()
    
    # Print statistics
    print("\nAveraged Results:")
    print(f"  - Number of simulations: {results['n_simulations']}")
    avg_disp_magnitudes = np.linalg.norm(results['average_displacements'], axis=1)
    print(f"  - Mean displacement magnitude: {np.mean(avg_disp_magnitudes):.4f}")
    print(f"  - Max displacement magnitude: {np.max(avg_disp_magnitudes):.4f}")
    
    print("="*60 + "\n")


def demo_2d_square_averaging():
    """Demonstrate multi-simulation averaging for 2D square lattice."""
    print("\n" + "="*60)
    print("DEMO: Multi-Simulation Averaging - 2D Square Lattice")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Network parameters
    network_params = {
        'size': 5,
        'mass': 1.0,
        'spring_constant': 8.0,
        'damping': 0.3,
        'dt': 0.01,
        'temperature': 1.5
    }
    
    # Run multiple simulations
    results = run_multiple_simulations(
        network_class=Network2DSquare,
        network_params=network_params,
        simulation_steps=1500,
        n_simulations=10,
        show_progress=True
    )
    
    # Visualize averaged results
    print("\nGenerating averaged visualization...")
    fig = visualize_averaged_results(
        results=results,
        network_params=network_params,
        title="2D Square Lattice - Averaged Over 10 Simulations"
    )
    
    filename = 'averaged_2d_square.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {filename}")
    plt.close()
    
    # Print statistics
    print("\nAveraged Results:")
    print(f"  - Number of simulations: {results['n_simulations']}")
    avg_disp_magnitudes = np.linalg.norm(results['average_displacements'], axis=1)
    print(f"  - Mean displacement magnitude: {np.mean(avg_disp_magnitudes):.4f}")
    print(f"  - Max displacement magnitude: {np.max(avg_disp_magnitudes):.4f}")
    
    print("="*60 + "\n")


def demo_2d_triangular_averaging():
    """Demonstrate multi-simulation averaging for 2D triangular lattice."""
    print("\n" + "="*60)
    print("DEMO: Multi-Simulation Averaging - 2D Triangular Lattice")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Network parameters
    network_params = {
        'size': 5,
        'mass': 1.0,
        'spring_constant': 8.0,
        'damping': 0.3,
        'dt': 0.01,
        'temperature': 1.5
    }
    
    # Run multiple simulations
    results = run_multiple_simulations(
        network_class=Network2DTriangular,
        network_params=network_params,
        simulation_steps=1500,
        n_simulations=10,
        show_progress=True
    )
    
    # Visualize averaged results
    print("\nGenerating averaged visualization...")
    fig = visualize_averaged_results(
        results=results,
        network_params=network_params,
        title="2D Triangular Lattice - Averaged Over 10 Simulations"
    )
    
    filename = 'averaged_2d_triangular.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {filename}")
    plt.close()
    
    # Print statistics
    print("\nAveraged Results:")
    print(f"  - Number of simulations: {results['n_simulations']}")
    avg_disp_magnitudes = np.linalg.norm(results['average_displacements'], axis=1)
    print(f"  - Mean displacement magnitude: {np.mean(avg_disp_magnitudes):.4f}")
    print(f"  - Max displacement magnitude: {np.max(avg_disp_magnitudes):.4f}")
    
    print("="*60 + "\n")


def main():
    """Run all averaging demos."""
    print("\n" + "*" * 60)
    print("QUANTUM GRAVITY SIMULATOR - MULTI-SIMULATION AVERAGING DEMOS")
    print("*" * 60)
    
    # Run all demos
    demo_1d_averaging()
    demo_2d_square_averaging()
    demo_2d_triangular_averaging()
    
    print("\n" + "*" * 60)
    print("All averaging demos completed successfully!")
    print("*" * 60 + "\n")


if __name__ == "__main__":
    main()
