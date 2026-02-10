"""
Example usage of the Quantum Gravity Simulator.

This script demonstrates how to use the simulator with different network configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_gravity_simulator import (
    Network1D, 
    Network2DSquare, 
    Network2DTriangular,
    visualize_network,
    print_results
)


def example_1d():
    """Example: 1D mass-spring chain."""
    print("\n" + "="*60)
    print("EXAMPLE 1: 1D Mass-Spring Chain")
    print("="*60)
    
    # Create a 1D network with 11 masses
    network = Network1D(
        n_masses=11,
        mass=1.0,
        spring_constant=10.0,
        damping=0.5,
        dt=0.01,
        temperature=2.0
    )
    
    print(f"Initial center node position: {network.positions[network.center_idx]}")
    
    # Run simulation
    print("Running simulation for 1000 steps...")
    network.simulate(steps=1000)
    
    print(f"Final center node position: {network.positions[network.center_idx]}")
    
    # Print results
    print_results(network)
    
    # Visualize
    fig = visualize_network(network, "1D Mass-Spring Chain")
    plt.savefig('quantum_gravity_1d.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: quantum_gravity_1d.png")
    plt.close()


def example_2d_square():
    """Example: 2D square lattice."""
    print("\n" + "="*60)
    print("EXAMPLE 2: 2D Square Lattice")
    print("="*60)
    
    # Create a 2D square network (5x5)
    network = Network2DSquare(
        size=5,
        mass=1.0,
        spring_constant=8.0,
        damping=0.3,
        dt=0.01,
        temperature=1.5
    )
    
    print(f"Initial center node position: {network.positions[network.center_idx]}")
    
    # Run simulation
    print("Running simulation for 1500 steps...")
    network.simulate(steps=1500)
    
    print(f"Final center node position: {network.positions[network.center_idx]}")
    
    # Print results
    print_results(network)
    
    # Visualize
    fig = visualize_network(network, "2D Square Lattice")
    plt.savefig('quantum_gravity_2d_square.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: quantum_gravity_2d_square.png")
    plt.close()


def example_2d_triangular():
    """Example: 2D triangular lattice."""
    print("\n" + "="*60)
    print("EXAMPLE 3: 2D Triangular Lattice")
    print("="*60)
    
    # Create a 2D triangular network (5x5)
    network = Network2DTriangular(
        size=5,
        mass=1.0,
        spring_constant=8.0,
        damping=0.3,
        dt=0.01,
        temperature=1.5
    )
    
    print(f"Initial center node position: {network.positions[network.center_idx]}")
    
    # Run simulation
    print("Running simulation for 1500 steps...")
    network.simulate(steps=1500)
    
    print(f"Final center node position: {network.positions[network.center_idx]}")
    
    # Print results
    print_results(network)
    
    # Visualize
    fig = visualize_network(network, "2D Triangular Lattice")
    plt.savefig('quantum_gravity_2d_triangular.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: quantum_gravity_2d_triangular.png")
    plt.close()


def main():
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print("QUANTUM GRAVITY SIMULATOR - EXAMPLES")
    print("*" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run examples
    example_1d()
    example_2d_square()
    example_2d_triangular()
    
    print("\n" + "*" * 60)
    print("All examples completed successfully!")
    print("*" * 60 + "\n")


if __name__ == "__main__":
    main()
