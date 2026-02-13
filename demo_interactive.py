"""
Demo script for interactive 3D surface visualization and live simulation.

This script demonstrates:
1. Interactive 3D surface plot that can be rotated and repositioned
2. Live simulation with real-time updates
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_gravity_simulator import (
    Network2DSquare,
    Network2DTriangular,
    visualize_network_3d_surface,
    animate_simulation_live
)


def demo_interactive_surface():
    """
    Demo: Create a network, simulate it, and show an interactive 3D surface plot.
    The user can rotate and reposition the view.
    """
    print("\n" + "="*70)
    print("DEMO 1: INTERACTIVE 3D SURFACE VISUALIZATION")
    print("="*70)
    print("\nThis demo will:")
    print("  1. Create a 5x5 square lattice network")
    print("  2. Run the simulation")
    print("  3. Display an INTERACTIVE 3D surface plot")
    print("  4. Allow you to rotate and zoom the plot with your mouse")
    print("="*70 + "\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a 2D square network
    print("Creating 2D square lattice...")
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
    print("\nRunning simulation for 1500 steps...")
    network.simulate(steps=1500, show_progress=True)
    
    print(f"Final center node position: {network.positions[network.center_idx]}")
    
    # Create interactive 3D surface visualization
    print("\nGenerating interactive 3D surface visualization...")
    fig = visualize_network_3d_surface(
        network, 
        title="Interactive 3D Surface - 2D Square Lattice",
        interactive=True
    )
    
    if fig is not None:
        plt.savefig('interactive_surface_demo.png', dpi=150, bbox_inches='tight')
        print("Saved static image to: interactive_surface_demo.png")
        plt.show()
        plt.close()
    
    print("\n" + "="*70)
    print("Demo 1 completed!")
    print("="*70 + "\n")


def demo_live_simulation():
    """
    Demo: Run a simulation with live updates showing the network state evolving in real-time.
    """
    print("\n" + "="*70)
    print("DEMO 2: LIVE SIMULATION WITH REAL-TIME UPDATES")
    print("="*70)
    print("\nThis demo will:")
    print("  1. Create a 5x5 triangular lattice network")
    print("  2. Run the simulation WITH LIVE UPDATES")
    print("  3. Update the 3D surface plot every 50 steps")
    print("  4. Allow you to watch the simulation evolve in real-time")
    print("="*70 + "\n")
    
    # Set random seed for reproducibility
    np.random.seed(123)
    
    # Parameters for the network
    network_params = {
        'size': 5,
        'mass': 1.0,
        'spring_constant': 8.0,
        'damping': 0.3,
        'dt': 0.01,
        'temperature': 1.5
    }
    
    # Run live simulation
    network = animate_simulation_live(
        network_class=Network2DTriangular,
        network_params=network_params,
        simulation_steps=1000,
        update_interval=50,
        title="Live Simulation - 2D Triangular Lattice"
    )
    
    # Save final state
    print("\nGenerating final state visualization...")
    fig = visualize_network_3d_surface(
        network,
        title="Final State - 2D Triangular Lattice",
        interactive=False
    )
    
    if fig is not None:
        plt.savefig('live_simulation_final.png', dpi=150, bbox_inches='tight')
        print("Saved final state to: live_simulation_final.png")
        plt.close()
    
    print("\n" + "="*70)
    print("Demo 2 completed!")
    print("="*70 + "\n")


def main():
    """Run all demos."""
    print("\n")
    print("*" * 70)
    print("INTERACTIVE 3D VISUALIZATION DEMOS")
    print("*" * 70)
    
    # Ask user which demo to run
    print("\nAvailable demos:")
    print("  1. Interactive 3D surface plot (you can rotate/zoom)")
    print("  2. Live simulation with real-time updates")
    print("  3. Both demos")
    
    choice = input("\nSelect demo (1/2/3) [3]: ").strip()
    
    if choice == '1':
        demo_interactive_surface()
    elif choice == '2':
        demo_live_simulation()
    else:
        # Run both by default
        demo_interactive_surface()
        demo_live_simulation()
    
    print("\n" + "*" * 70)
    print("All demos completed successfully!")
    print("*" * 70 + "\n")


if __name__ == "__main__":
    main()
