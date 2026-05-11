#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:14:14 2026

@author: benvarcoe
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from quantum_gravity_simulator import (
    Network1D,
    Network2DSquare,
    Network2DTriangular,
    run_multiple_simulations,
    visualize_averaged_results,
    visualize_network_3d,
    visualize_network_3d_surface
)


# Define network parameters
network_params = {
    'size': 13,
    'mass': 1.0,
    'spring_constant': 1.0,
    'damping': 0.1,
    'dt': 0.01,
    'temperature': 50
}

# Simulation parameters
network_class = Network2DSquare
simulation_steps = 1500
n_simulations = 60

# Create figures directory if it doesn't exist
figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)

# Run simulations (no seed set - each simulation is unique)
results = run_multiple_simulations(
    network_class=network_class,
    network_params=network_params,
    simulation_steps=simulation_steps,
    n_simulations=n_simulations,
    show_progress=True
)

# Create simulation conditions text
conditions_text = (
    f"Simulation Conditions:\n"
    f"  size: {network_params['size']}\n"
    f"  mass: {network_params['mass']}\n"
    f"  spring_constant: {network_params['spring_constant']}\n"
    f"  damping: {network_params['damping']}\n"
    f"  dt: {network_params['dt']}\n"
    f"  temperature: {network_params['temperature']}\n"
    f"  network_class: {network_class.__name__}\n"
    f"  simulation_steps: {simulation_steps}\n"
    f"  n_simulations: {n_simulations}"
)

# Visualize averaged results
fig = visualize_averaged_results(
    results=results,
    network_class=network_class,
    network_params=network_params,
    title="Averaged Simulation Results"
)

# Add simulation conditions as text box to the figure
# Position at top-left with slightly larger font for better readability
fig.text(0.01, 0.99, conditions_text, transform=fig.transFigure, 
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, pad=0.5))

# Access averaged data
average_positions = results['average_final_positions']
average_displacements = results['average_displacements']
std_displacements = results['std_displacements']

# Generate incremental filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_filename = 'averaged_2d_square'
filename = os.path.join(figures_dir, f'{base_filename}_{timestamp}.png')

# Save figure
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"Saved visualization to: {filename}")
plt.close()

# Generate 3D surface plot visualization from a representative simulation
print("\nGenerating 3D displacement surface plot visualization...")
# Run a single simulation to get a network object for 3D visualization
representative_network = network_class(**network_params)
representative_network.simulate(steps=simulation_steps, show_progress=False)

# Create 3D surface plot visualization
fig_3d = visualize_network_3d_surface(representative_network, 
                                      f"{network_class.__name__} - 3D Surface Plot",
                                      interactive=False)
if fig_3d:
    # Save 3D surface plot with same timestamp
    filename_3d = os.path.join(figures_dir, f'{base_filename}_3d_surface_{timestamp}.png')
    fig_3d.savefig(filename_3d, dpi=150, bbox_inches='tight')
    print(f"Saved 3D surface plot to: {filename_3d}")
    plt.close(fig_3d)

# Print statistics
print("\nAveraged Results:")
print(f"  - Number of simulations: {results['n_simulations']}")
avg_disp_magnitudes = np.linalg.norm(results['average_displacements'], axis=1)
print(f"  - Mean displacement magnitude: {np.mean(avg_disp_magnitudes):.4f}")
print(f"  - Max displacement magnitude: {np.max(avg_disp_magnitudes):.4f}")

print("="*60 + "\n")