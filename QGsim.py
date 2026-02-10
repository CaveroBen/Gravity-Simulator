#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:14:14 2026

@author: benvarcoe
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


# Define network parameters
network_params = {
    'size': 13,
    'mass': 1.0,
    'spring_constant': 1.0,
    'damping': 0.1,
    'dt': 0.01,
    'temperature': 50
}

# Run 10 simulations
results = run_multiple_simulations(
    network_class=Network2DSquare,
    network_params=network_params,
    simulation_steps=1500,
    n_simulations=60,
    show_progress=True
)

# Visualize averaged results
fig = visualize_averaged_results(
    results=results,
    network_class=Network2DSquare,
    network_params=network_params,
    title="Averaged Simulation Results"
)

# Access averaged data
average_positions = results['average_final_positions']
average_displacements = results['average_displacements']
std_displacements = results['std_displacements']
    
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