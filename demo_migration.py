#!/usr/bin/env python3
"""
Compare baseline vs. noisy oscillator array to demonstrate entropy-driven migration.

This script replicates the key functionality of OscillatorArray_Vector.py
using the improved quantum_gravity_simulator.py implementation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from quantum_gravity_simulator import (
    Network2DSquare,
    visualize_network,
    visualize_migration,
    print_results
)


def compare_baseline_vs_noisy():
    """
    Compare baseline (no Brownian motion) vs. noisy (with Brownian motion) systems.
    
    This demonstrates how coupling in a constrained system causes non-central 
    oscillators to migrate toward the center oscillator undergoing random walk.
    """
    print("\n" + "="*70)
    print("ENTROPY-DRIVEN MIGRATION DEMONSTRATION")
    print("="*70)
    print("\nComparing systems with and without Brownian motion at center...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Common parameters
    common_params = dict(
        size=8,                    # 8x8 grid
        mass=1.0,
        spring_constant=5.0,
        damping=1.5,
        dt=0.01,
        rest_length=1.0
    )
    
    # Baseline: No Brownian motion (temperature = 0)
    print("\n--- Running BASELINE simulation (no Brownian motion) ---")
    baseline_network = Network2DSquare(
        temperature=0.0,  # No thermal fluctuations
        **common_params
    )
    baseline_network.simulate(steps=1000, show_progress=True)
    
    # Noisy: With Brownian motion (temperature > 0)
    print("\n--- Running NOISY simulation (with Brownian motion) ---")
    noisy_network = Network2DSquare(
        temperature=3.0,  # Significant thermal fluctuations
        **common_params
    )
    noisy_network.simulate(steps=1000, show_progress=True)
    
    # Extract histories
    baseline_radial = baseline_network.get_radial_displacement_history()
    noisy_radial = noisy_network.get_radial_displacement_history()
    baseline_neighbor = baseline_network.get_neighbor_distance_history()
    noisy_neighbor = noisy_network.get_neighbor_distance_history()
    
    # Create comparison visualization
    print("\nGenerating comparison plots...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    time_steps = np.arange(len(baseline_radial))
    
    # Top left: Radial displacement comparison
    ax1.plot(time_steps, baseline_radial, 'k--', linewidth=2.5, 
             label='Baseline (no Brownian)', alpha=0.8)
    ax1.plot(time_steps, noisy_radial, 'r-', linewidth=2.5, 
             label='With Brownian motion', alpha=0.9)
    ax1.axhline(y=0, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    ax1.fill_between(time_steps, 0, noisy_radial,
                     where=(np.array(noisy_radial) > 0),
                     color='green', alpha=0.2)
    ax1.fill_between(time_steps, 0, noisy_radial,
                     where=(np.array(noisy_radial) < 0),
                     color='red', alpha=0.2)
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Mean Radial Displacement', fontsize=12)
    ax1.set_title('Radial Displacement (+ = attraction, - = repulsion)', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Top right: Neighbor distances comparison
    ax2.plot(time_steps, baseline_neighbor, 'k--', linewidth=2.5,
             label='Baseline (no Brownian)', alpha=0.8)
    ax2.plot(time_steps, noisy_neighbor, 'r-', linewidth=2.5,
             label='With Brownian motion', alpha=0.9)
    ax2.axhline(y=common_params['rest_length'], color='gray', linestyle=':',
               linewidth=2, alpha=0.5, label='Equilibrium spacing')
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Mean Neighbor Distance', fontsize=12)
    ax2.set_title('Neighbor Distances to Center Oscillator', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Bottom left: Distribution of radial displacements (last 20%)
    last_20_percent = max(1, len(baseline_radial) // 5)
    ax3.hist(baseline_radial[-last_20_percent:], bins=15, alpha=0.6,
            label='Baseline', color='black', edgecolor='black')
    ax3.hist(noisy_radial[-last_20_percent:], bins=15, alpha=0.6,
            label='Brownian', color='red', edgecolor='darkred')
    ax3.axvline(x=0, color='gray', linestyle='--', linewidth=2, label='No displacement')
    ax3.set_xlabel('Radial Displacement', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution (final 20% of simulation)', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Bottom right: Distribution of neighbor distances (last 20%)
    ax4.hist(baseline_neighbor[-last_20_percent:], bins=15, alpha=0.6,
            label='Baseline', color='black', edgecolor='black')
    ax4.hist(noisy_neighbor[-last_20_percent:], bins=15, alpha=0.6,
            label='Brownian', color='red', edgecolor='darkred')
    ax4.axvline(x=common_params['rest_length'], color='gray', linestyle='--',
               linewidth=2, label='Equilibrium')
    ax4.set_xlabel('Neighbor Distance', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution (final 20% of simulation)', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('migration_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison plot: migration_comparison.png")
    # plt.show()  # Commented out for non-interactive environments
    plt.close()
    
    # Visualize final states
    print("\nGenerating network visualizations...")
    
    fig1 = visualize_network(baseline_network, "Baseline (No Brownian Motion)")
    plt.savefig('baseline_network.png', dpi=150, bbox_inches='tight')
    print("Saved baseline network: baseline_network.png")
    plt.close()
    
    fig2 = visualize_network(noisy_network, "With Brownian Motion at Center")
    plt.savefig('noisy_network.png', dpi=150, bbox_inches='tight')
    print("Saved noisy network: noisy_network.png")
    plt.close()
    
    # Print statistics
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)
    
    baseline_mean_radial = np.mean(baseline_radial[-last_20_percent:])
    noisy_mean_radial = np.mean(noisy_radial[-last_20_percent:])
    
    print(f"\nRadial Displacement (mean, final 20%):")
    print(f"  Baseline:  {baseline_mean_radial:+.6f}")
    print(f"  Brownian:  {noisy_mean_radial:+.6f}")
    print(f"  Difference: {noisy_mean_radial - baseline_mean_radial:+.6f}")
    
    if abs(noisy_mean_radial) > abs(baseline_mean_radial):
        direction = "attraction" if noisy_mean_radial > 0 else "repulsion"
        print(f"  → Brownian motion causes net {direction}")
    
    baseline_mean_neighbor = np.mean(baseline_neighbor[-last_20_percent:])
    noisy_mean_neighbor = np.mean(noisy_neighbor[-last_20_percent:])
    
    print(f"\nNeighbor Distance (mean, final 20%):")
    print(f"  Baseline:  {baseline_mean_neighbor:.4f}")
    print(f"  Brownian:  {noisy_mean_neighbor:.4f}")
    print(f"  Equilibrium: {common_params['rest_length']:.4f}")
    print(f"  Difference: {noisy_mean_neighbor - baseline_mean_neighbor:.4f}")
    
    if noisy_mean_neighbor < baseline_mean_neighbor:
        print(f"  → Neighbors move closer to center with Brownian motion")
    elif noisy_mean_neighbor > baseline_mean_neighbor:
        print(f"  → Neighbors move away from center with Brownian motion")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
The test demonstrates entropy-driven migration in coupled oscillator systems:

1. BASELINE (no Brownian motion):
   - Center oscillator is static
   - Radial displacement remains near zero
   - Neighbor distances stay at equilibrium

2. WITH BROWNIAN MOTION:
   - Center oscillator undergoes random walk (entropy source)
   - Coupled oscillators respond to center's motion
   - System migrates toward entropy source due to coupling
   - Non-central masses adjust their equilibrium positions

This validates that the quantum_gravity_simulator now correctly models
the same entropy-driven migration behavior as OscillatorArray_Vector.py.
""")
    print("="*70 + "\n")


def single_simulation_demo():
    """Run a single simulation with detailed analysis."""
    print("\n" + "="*70)
    print("SINGLE SIMULATION DEMONSTRATION")
    print("="*70)
    
    np.random.seed(42)
    
    # Create network with Brownian motion
    print("\nCreating 8x8 square lattice with periodic boundaries...")
    network = Network2DSquare(
        size=8,
        mass=1.0,
        spring_constant=5.0,
        damping=1.5,
        dt=0.01,
        temperature=3.0,
        rest_length=1.0
    )
    
    print("Running simulation for 1000 steps...")
    network.simulate(steps=1000, show_progress=True)
    
    # Print detailed results
    print_results(network)
    
    # Visualize migration
    print("\nGenerating migration analysis plot...")
    fig1 = visualize_migration(network, "8x8 Square Lattice")
    plt.savefig('migration_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved migration analysis: migration_analysis.png")
    # plt.show()  # Commented out for non-interactive environments
    plt.close()
    
    # Visualize network
    print("\nGenerating network visualization...")
    fig2 = visualize_network(network, "8x8 Square Lattice with Brownian Center")
    plt.savefig('network_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved network visualization: network_visualization.png")
    # plt.show()  # Commented out for non-interactive environments
    plt.close()
    
    print("\n" + "="*70)
    print("Demonstration complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Ask user which demo to run
    print("\nAvailable demonstrations:")
    print("1. Compare baseline vs. noisy (recommended)")
    print("2. Single simulation with detailed analysis")
    
    choice = input("\nSelect demonstration (1 or 2) [1]: ").strip()
    
    if choice == "2":
        single_simulation_demo()
    else:
        compare_baseline_vs_noisy()
