#!/usr/bin/env python3
"""
Demo script showing the new tilt and rotate features for 3D visualizations.

This demonstrates how to use the elev and azim parameters to view 3D plots
from different angles.
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_gravity_simulator import (
    Network2DSquare,
    visualize_network_3d_surface
)


def demo_different_angles():
    """
    Demo: Create a network and visualize it from different angles.
    """
    print("\n" + "="*70)
    print("DEMO: TILT AND ROTATE FEATURES FOR 3D PLOTS")
    print("="*70)
    print("\nThis demo will:")
    print("  1. Create and simulate a 2D square lattice")
    print("  2. Generate 3D surface plots from different viewing angles")
    print("  3. Show how to use elev and azim parameters")
    print("="*70 + "\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create and simulate a 2D square network
    print("Creating and simulating 2D square lattice...")
    network = Network2DSquare(
        size=7,
        mass=1.0,
        spring_constant=8.0,
        damping=0.3,
        dt=0.01,
        temperature=2.0
    )
    
    network.simulate(steps=1000, show_progress=True)
    
    # Define different viewing angles to demonstrate
    angles = [
        {"name": "Default View", "elev": 20, "azim": 45, 
         "desc": "Standard angle - good for general viewing"},
        {"name": "Top View", "elev": 90, "azim": 0,
         "desc": "Bird's eye view - shows spatial distribution"},
        {"name": "Side View", "elev": 0, "azim": 0,
         "desc": "Profile view - emphasizes height differences"},
        {"name": "Low Angle", "elev": 10, "azim": 135,
         "desc": "Low angle - dramatic perspective"},
        {"name": "High Angle", "elev": 60, "azim": 225,
         "desc": "High angle - better depth perception"},
    ]
    
    print("\nGenerating visualizations from different angles...")
    print("-" * 70)
    
    for i, angle_config in enumerate(angles, 1):
        name = angle_config["name"]
        elev = angle_config["elev"]
        azim = angle_config["azim"]
        desc = angle_config["desc"]
        
        print(f"\n{i}. {name}")
        print(f"   Elevation: {elev}°, Azimuth: {azim}°")
        print(f"   {desc}")
        
        # Create the visualization
        fig = visualize_network_3d_surface(
            network,
            title=f"{name} (elev={elev}°, azim={azim}°)",
            elev=elev,
            azim=azim,
            interactive=False  # Set to True for interactive rotation
        )
        
        if fig:
            filename = f'3d_view_{i}_{name.lower().replace(" ", "_")}.png'
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"   ✓ Saved to: {filename}")
            plt.close(fig)
    
    print("\n" + "="*70)
    print("USAGE EXAMPLES")
    print("="*70)
    print("\nBasic usage with custom viewing angle:")
    print("```python")
    print("fig = visualize_network_3d_surface(")
    print("    network,")
    print("    title='My Custom View',")
    print("    elev=30,    # Elevation angle in degrees")
    print("    azim=120,   # Azimuth angle in degrees")
    print("    interactive=True  # Enable mouse rotation")
    print(")")
    print("plt.show()")
    print("```")
    
    print("\nCommon viewing angles:")
    print("  - Top view:     elev=90, azim=0")
    print("  - Side view:    elev=0,  azim=0")
    print("  - Front view:   elev=0,  azim=90")
    print("  - Default:      elev=20, azim=45")
    print("  - Isometric:    elev=35.264, azim=45")
    
    print("\n" + "="*70)
    print("Demo completed! Check the generated PNG files.")
    print("="*70 + "\n")


if __name__ == "__main__":
    demo_different_angles()
