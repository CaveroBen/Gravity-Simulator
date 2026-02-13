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
    visualize_network_3d,
    print_results
)


def get_input_with_default(prompt: str, default_value, value_type=float):
    """
    Get user input with a default value.
    
    Args:
        prompt: The prompt to display
        default_value: The default value to use if user presses enter
        value_type: The type to convert the input to (float or int)
    
    Returns:
        The user input value or default value
    """
    user_input = input(f"{prompt} [{default_value}]: ").strip()
    if user_input == "":
        return default_value
    try:
        return value_type(user_input)
    except ValueError:
        print(f"Invalid input. Using default: {default_value}")
        return default_value


def interactive_simulation():
    """Run an interactive simulation with user-configurable parameters."""
    print("\n" + "="*60)
    print("INTERACTIVE QUANTUM GRAVITY SIMULATION")
    print("="*60)
    print("\nPress Enter to accept default values for each parameter.\n")
    
    # Get simulation parameters with defaults
    network_type = input("Network type (1d/square/triangular) [square]: ").strip().lower()
    if network_type not in ['1d', 'square', 'triangular']:
        network_type = 'square'
    
    if network_type == '1d':
        array_size = get_input_with_default("Number of masses", 11, int)
    else:
        array_size = get_input_with_default("Array size (NxN grid)", 5, int)
    
    simulation_time = get_input_with_default("Simulation time steps", 1500, int)
    temperature = get_input_with_default("Thermal excitation (temperature)", 1.5, float)
    spring_constant = get_input_with_default("Spring constant", 8.0, float)
    
    # Create network based on type
    print(f"\nCreating {network_type} network...")
    if network_type == '1d':
        network = Network1D(
            n_masses=array_size,
            mass=1.0,
            spring_constant=spring_constant,
            damping=0.3,
            dt=0.01,
            temperature=temperature
        )
        title = f"1D Chain ({array_size} masses)"
    elif network_type == 'triangular':
        network = Network2DTriangular(
            size=array_size,
            mass=1.0,
            spring_constant=spring_constant,
            damping=0.3,
            dt=0.01,
            temperature=temperature
        )
        title = f"2D Triangular Lattice ({array_size}x{array_size})"
    else:  # square
        network = Network2DSquare(
            size=array_size,
            mass=1.0,
            spring_constant=spring_constant,
            damping=0.3,
            dt=0.01,
            temperature=temperature
        )
        title = f"2D Square Lattice ({array_size}x{array_size})"
    
    print(f"Initial center node position: {network.positions[network.center_idx]}")
    
    # Run simulation with progress bar
    print(f"\nRunning simulation for {simulation_time} steps...")
    network.simulate(steps=simulation_time, show_progress=True)
    
    print(f"Final center node position: {network.positions[network.center_idx]}")
    
    # Comment out detailed results printing as requested
    # print_results(network)
    
    # Visualize
    print("\nGenerating visualization...")
    fig = visualize_network(network, title)
    filename = f'quantum_gravity_{network_type}_interactive.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {filename}")
    plt.show()
    plt.close()
    
    # Generate 3D visualization if it's a 2D network
    if network_type in ['square', 'triangular']:
        print("\nGenerating 3D displacement visualization...")
        fig_3d = visualize_network_3d(network, f"{title} - 3D View")
        if fig_3d is not None:
            filename_3d = f'quantum_gravity_{network_type}_3d_interactive.png'
            plt.savefig(filename_3d, dpi=150, bbox_inches='tight')
            print(f"Saved 3D visualization to: {filename_3d}")
            plt.show()
            plt.close()
    
    print("\n" + "="*60)
    print("Simulation completed!")
    print("="*60 + "\n")


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
    
    # Run simulation with progress bar
    print("Running simulation for 1000 steps...")
    network.simulate(steps=1000, show_progress=True)
    
    print(f"Final center node position: {network.positions[network.center_idx]}")
    
    # Print results - commented out as requested
    # print_results(network)
    
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
    
    # Run simulation with progress bar
    print("Running simulation for 1500 steps...")
    network.simulate(steps=1500, show_progress=True)
    
    print(f"Final center node position: {network.positions[network.center_idx]}")
    
    # Print results - commented out as requested
    # print_results(network)
    
    # Visualize
    fig = visualize_network(network, "2D Square Lattice")
    plt.savefig('quantum_gravity_2d_square.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: quantum_gravity_2d_square.png")
    plt.close()
    
    # 3D visualization
    fig_3d = visualize_network_3d(network, "2D Square Lattice - 3D View")
    if fig_3d is not None:
        plt.savefig('quantum_gravity_2d_square_3d.png', dpi=150, bbox_inches='tight')
        print("Saved 3D visualization to: quantum_gravity_2d_square_3d.png")
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
    
    # Run simulation with progress bar
    print("Running simulation for 1500 steps...")
    network.simulate(steps=1500, show_progress=True)
    
    print(f"Final center node position: {network.positions[network.center_idx]}")
    
    # Print results - commented out as requested
    # print_results(network)
    
    # Visualize
    fig = visualize_network(network, "2D Triangular Lattice")
    plt.savefig('quantum_gravity_2d_triangular.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: quantum_gravity_2d_triangular.png")
    plt.close()
    
    # 3D visualization
    fig_3d = visualize_network_3d(network, "2D Triangular Lattice - 3D View")
    if fig_3d is not None:
        plt.savefig('quantum_gravity_2d_triangular_3d.png', dpi=150, bbox_inches='tight')
        print("Saved 3D visualization to: quantum_gravity_2d_triangular_3d.png")
        plt.close()


def main():
    """Run all examples or interactive mode."""
    print("\n")
    print("*" * 60)
    print("QUANTUM GRAVITY SIMULATOR - EXAMPLES")
    print("*" * 60)
    
    # Ask user which mode to run
    mode = input("\nRun mode (interactive/batch) [interactive]: ").strip().lower()
    if mode not in ['batch', 'b']:
        # Run interactive mode
        interactive_simulation()
    else:
        # Run batch examples
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
