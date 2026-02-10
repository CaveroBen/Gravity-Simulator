#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 15:07:25 2026

@author: benvarcoe
"""

# 2D_lattice_entropy_emergence.py
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 2D Lattice with periodic boundary conditions
# -----------------------------
class Lattice2D:
    def __init__(self, Nx, Ny, spacing=1.0, seed=0):
        """
        Nx, Ny: grid dimensions
        spacing: equilibrium distance between lattice sites
        """
        self.Nx = Nx
        self.Ny = Ny
        self.N = Nx * Ny
        self.spacing = spacing
        self.rng = np.random.default_rng()
        
        # Initialize positions on regular grid
        x = np.arange(Nx) * spacing
        y = np.arange(Ny) * spacing
        xx, yy = np.meshgrid(x, y, indexing='ij')
        self.positions = np.stack([xx.ravel(), yy.ravel()], axis=1)
        self.initial_positions = self.positions.copy()
        
        # Initialize random phases
        self.thetas = self.rng.uniform(0, 2*np.pi, size=self.N)
        
        # Initialize velocities
        self.velocities = np.zeros_like(self.positions)
        
        # Build neighbor list and grid indices
        self.neighbors = self._build_neighbor_list()
        self.grid_indices = self._build_grid_indices()
        
    def _build_neighbor_list(self):
        """Build list of nearest neighbors with periodic boundaries"""
        neighbors = []
        for i in range(self.Nx):
            for j in range(self.Ny):
                idx = i * self.Ny + j
                # Right, left, up, down with wrapping
                right = ((i + 1) % self.Nx) * self.Ny + j
                left = ((i - 1) % self.Nx) * self.Ny + j
                up = i * self.Ny + ((j + 1) % self.Ny)
                down = i * self.Ny + ((j - 1) % self.Ny)
                neighbors.append([right, left, up, down])
        return neighbors
    
    def _build_grid_indices(self):
        """Map flat index to (i,j) grid coordinates"""
        indices = []
        for i in range(self.Nx):
            for j in range(self.Ny):
                indices.append((i, j))
        return indices
    
    def get_neighbor_indices(self, i):
        """Get the 4 nearest neighbors of particle i"""
        return self.neighbors[i]
    
    def periodic_vector(self, i, j):
        """Get minimum vector from i to j considering periodic boundaries"""
        dx = self.positions[j] - self.positions[i]
        # Apply periodic boundary conditions
        Lx = self.Nx * self.spacing
        Ly = self.Ny * self.spacing
        dx[0] = dx[0] - Lx * np.round(dx[0] / Lx)
        dx[1] = dx[1] - Ly * np.round(dx[1] / Ly)
        return dx

# -----------------------------
# Lattice springs (only mechanical force)
# -----------------------------
def lattice_spring_force(lattice, k_spring):
    """
    Harmonic springs connecting nearest neighbors.
    This provides the basic mechanical structure, nothing entropic.
    """
    F = np.zeros_like(lattice.positions)
    rest_len = lattice.spacing
    
    for i in range(lattice.N):
        for j in lattice.get_neighbor_indices(i):
            r_vec = lattice.periodic_vector(i, j)
            dist = np.linalg.norm(r_vec)
            if dist > 1e-12:
                # Simple Hookean spring: F = -k(r - r0)
                F[i] += k_spring * (dist - rest_len) * (r_vec / dist)
    
    return F

# -----------------------------
# Kuramoto phase dynamics
# -----------------------------
def theta_dynamics_lattice(lattice, omega, K_theta):
    """
    Phase dynamics with nearest neighbor coupling.
    """
    N = lattice.N
    theta_dot = np.copy(omega)
    
    for i in range(N):
        coupling = 0.0
        for j in lattice.get_neighbor_indices(i):
            # Standard Kuramoto coupling
            coupling += np.sin(lattice.thetas[j] - lattice.thetas[i])
        theta_dot[i] += K_theta * coupling / 4.0
    
    return theta_dot

# -----------------------------
# Distance analysis tools
# -----------------------------
def compute_neighbor_distances(lattice, noisy_indices):
    """
    Compute distances from noisy oscillators to their neighbors.
    Returns mean distance.
    """
    neighbor_dists = []
    
    for idx in noisy_indices:
        for neighbor_idx in lattice.get_neighbor_indices(idx):
            r_vec = lattice.periodic_vector(neighbor_idx, idx)
            dist = np.linalg.norm(r_vec)
            neighbor_dists.append(dist)
    
    return np.mean(neighbor_dists) if neighbor_dists else lattice.spacing

def compute_displacement_map(lattice):
    """Compute how far each oscillator has moved from initial position"""
    displacements = np.zeros(lattice.N)
    displacement_vectors = np.zeros((lattice.N, 2))
    
    for i in range(lattice.N):
        # Need to account for periodic boundaries
        dx = lattice.positions[i] - lattice.initial_positions[i]
        Lx = lattice.Nx * lattice.spacing
        Ly = lattice.Ny * lattice.spacing
        dx[0] = dx[0] - Lx * np.round(dx[0] / Lx)
        dx[1] = dx[1] - Ly * np.round(dx[1] / Ly)
        displacements[i] = np.linalg.norm(dx)
        displacement_vectors[i] = dx
    
    return displacements, displacement_vectors

def compute_radial_displacement(lattice, noisy_indices):
    """
    Compute mean displacement toward/away from noisy oscillator.
    Positive = toward noisy, Negative = away from noisy
    """
    if not noisy_indices:
        return 0.0
    
    radial_disps = []
    noisy_idx = noisy_indices[0]
    noisy_pos = lattice.positions[noisy_idx]
    noisy_initial = lattice.initial_positions[noisy_idx]
    
    for i in range(lattice.N):
        if i in noisy_indices:
            continue
        
        # Vector from oscillator i to noisy oscillator (current and initial)
        r_current = lattice.periodic_vector(i, noisy_idx)
        
        # Initial vector (from initial positions)
        dx = noisy_initial - lattice.initial_positions[i]
        Lx = lattice.Nx * lattice.spacing
        Ly = lattice.Ny * lattice.spacing
        dx[0] = dx[0] - Lx * np.round(dx[0] / Lx)
        dx[1] = dx[1] - Ly * np.round(dx[1] / Ly)
        r_initial = dx
        
        # Radial displacement: positive means moved toward noisy
        dist_initial = np.linalg.norm(r_initial)
        dist_current = np.linalg.norm(r_current)
        radial_disp = dist_initial - dist_current
        radial_disps.append(radial_disp)
    
    return np.mean(radial_disps)

# -----------------------------
# Simulation runner
# -----------------------------
def run_lattice_sim(
    Nx=10, Ny=10, spacing=1.0,
    total_time=100.0, dt=0.001,
    mass=1.0, gamma=1.5,
    K_theta=0.8, omega_mean=1.0, omega_spread=0.05,
    noisy_indices=None, 
    sigma_position_noisy=0.0,  # Spatial noise for noisy oscillator
    sigma_theta_noisy=0.0,     # Phase noise for noisy oscillator
    sigma_theta_others=0.0,    # Phase noise for others
    k_spring=2.0,
    seed=1
):
    """
    Run 2D lattice simulation with NO explicit entropic forces.
    
    The noisy oscillator undergoes Brownian motion (spatial noise).
    We measure if its neighbors' equilibrium positions shift.
    """
    lattice = Lattice2D(Nx, Ny, spacing)
    steps = int(total_time / dt)
    
    # Set natural frequencies
    rng = np.random.default_rng()
    omega = rng.normal(omega_mean, omega_spread, size=lattice.N)
    
    # Select noisy oscillators (default: center oscillator)
    if noisy_indices is None:
        center_i = Nx // 2
        center_j = Ny // 2
        center_idx = center_i * Ny + center_j
        noisy_indices = [center_idx]
    
    print(f"Lattice: {Nx}x{Ny} = {lattice.N} oscillators")
    print(f"Noisy oscillators: {len(noisy_indices)} at indices {noisy_indices}")
    print(f"Spatial noise σ_pos = {sigma_position_noisy:.3f}")
    print(f"Running for {total_time}s with dt={dt}...")
    
    # Storage
    history = {
        't': [],
        'positions': [],
        'thetas': [],
        'neighbor_dist': [],
        'sync_order': [],
        'displacements': [],
        'displacement_vectors': [],
        'radial_displacement': [],
        'noisy_positions': []
    }
    
    for step in range(steps):
        # ONLY mechanical spring forces (no entropic force!)
        F_spr = lattice_spring_force(lattice, k_spring)
        
        # Update positions (damped dynamics)
        accel = (F_spr - gamma * lattice.velocities) / mass
        lattice.velocities += accel * dt
        lattice.positions += lattice.velocities * dt
        
        # ADD BROWNIAN MOTION to noisy oscillator(s)
        if sigma_position_noisy > 0:
            for idx in noisy_indices:
                # Spatial random walk (Brownian motion)
                dW_pos = rng.normal(0.0, sigma_position_noisy * np.sqrt(dt), size=2)
                lattice.positions[idx] += dW_pos
        
        # Update phases with phase noise
        theta_dot = theta_dynamics_lattice(lattice, omega, K_theta)
        noise = np.full(lattice.N, sigma_theta_others)
        noise[noisy_indices] = sigma_theta_noisy
        dW_theta = rng.normal(0.0, np.sqrt(dt), size=lattice.N)
        lattice.thetas += theta_dot * dt + noise * dW_theta
        lattice.thetas %= (2 * np.pi)
        
        # Store data every 200 steps
        if step % 200 == 0:
            history['t'].append(step * dt)
            history['positions'].append(lattice.positions.copy())
            history['thetas'].append(lattice.thetas.copy())
            
            # Track noisy oscillator position
            if noisy_indices:
                history['noisy_positions'].append(lattice.positions[noisy_indices[0]].copy())
            
            # Measure neighbor distances to noisy oscillators
            neighbor_d = compute_neighbor_distances(lattice, noisy_indices)
            history['neighbor_dist'].append(neighbor_d)
            
            # Synchronization order
            R = np.abs(np.mean(np.exp(1j * lattice.thetas)))
            history['sync_order'].append(R)
            
            # Displacement from initial positions
            disp, disp_vec = compute_displacement_map(lattice)
            history['displacements'].append(disp.copy())
            history['displacement_vectors'].append(disp_vec.copy())
            
            # Radial displacement toward/away from noisy oscillator
            radial_d = compute_radial_displacement(lattice, noisy_indices)
            history['radial_displacement'].append(radial_d)
    
    history['noisy_indices'] = noisy_indices
    history['lattice'] = lattice
    return history

# -----------------------------
# Visualization
# -----------------------------
def visualize_emergence(history, frame=-1):
    """Visualize emergent effects with vector field"""
    lattice = history['lattice']
    pos = history['positions'][frame]
    thetas = history['thetas'][frame]
    noisy = history['noisy_indices']
    displacements = history['displacements'][frame]
    displacement_vectors = history['displacement_vectors'][frame]
    
    fig = plt.figure(figsize=(18, 5))
    
    # Panel 1: Vector field of displacements
    ax1 = plt.subplot(131)
    
    # Color by displacement magnitude
    scatter1 = ax1.scatter(pos[:, 0], pos[:, 1], 
                          c=displacements, cmap='plasma', s=80,
                          vmin=0, vmax=np.percentile(displacements, 98),
                          edgecolors='black', linewidths=0.5, zorder=3)
    
    # Draw displacement vectors
    skip = 1  # Draw every nth vector
    scale = 15  # Scale factor for arrow length
    for i in range(0, lattice.N, skip):
        if i not in noisy:  # Don't draw vector for noisy oscillator
            dx, dy = displacement_vectors[i]
            if np.sqrt(dx**2 + dy**2) > 1e-6:  # Only draw if non-zero
                ax1.arrow(pos[i, 0], pos[i, 1], dx*scale, dy*scale,
                         head_width=0.08, head_length=0.12, fc='blue', ec='blue',
                         alpha=0.6, linewidth=1.5, zorder=2)
    
    # Highlight noisy oscillator
    ax1.scatter(pos[noisy, 0], pos[noisy, 1], 
               s=300, facecolors='none', edgecolors='red', linewidths=3, zorder=4)
    
    # Draw initial lattice structure faintly
    initial_pos = lattice.initial_positions
    for i in range(lattice.N):
        for j in lattice.get_neighbor_indices(i):
            if j > i:
                ax1.plot([initial_pos[i,0], initial_pos[j,0]], 
                        [initial_pos[i,1], initial_pos[j,1]], 
                        'k-', alpha=0.1, linewidth=0.5, zorder=1)
    
    ax1.set_aspect('equal')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title(f'Displacement Vectors (t={history["t"][frame]:.1f})', fontsize=12)
    plt.colorbar(scatter1, ax=ax1, label='|Displacement|')
    
    # Panel 2: Trajectory of noisy oscillator
    ax2 = plt.subplot(132)
    if history['noisy_positions']:
        traj = np.array(history['noisy_positions'])
        ax2.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.3, linewidth=1)
        ax2.scatter(traj[::10, 0], traj[::10, 1], c=np.arange(0, len(traj), 10), 
                   cmap='cool', s=20, alpha=0.6)
        ax2.scatter(traj[0, 0], traj[0, 1], c='green', s=150, marker='o', 
                   edgecolors='darkgreen', linewidths=2, label='Start', zorder=5)
        ax2.scatter(traj[-1, 0], traj[-1, 1], c='red', s=150, marker='s', 
                   edgecolors='darkred', linewidths=2, label='End', zorder=5)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_title('Brownian Trajectory of Noisy Oscillator', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Time series of radial displacement
    ax3 = plt.subplot(133)
    ax3.plot(history['t'], history['radial_displacement'], 'b-', linewidth=2.5, 
            label='Mean radial displacement')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2, label='No net displacement')
    ax3.fill_between(history['t'], 0, history['radial_displacement'], 
                     where=(np.array(history['radial_displacement']) > 0),
                     color='green', alpha=0.3, label='Attraction')
    ax3.fill_between(history['t'], 0, history['radial_displacement'], 
                     where=(np.array(history['radial_displacement']) < 0),
                     color='red', alpha=0.3, label='Repulsion')
    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_ylabel('Mean displacement toward noisy osc.', fontsize=11)
    ax3.set_title('Emergent Attraction/Repulsion', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_baseline_vs_noisy():
    """Compare baseline (no noise) vs Brownian noisy oscillator"""
    common_params = dict(
        Nx=12, Ny=12, spacing=1.0,
        total_time=50.0, dt=0.001,
        mass=1.0, gamma=2.0,
        K_theta=0.0006,
        k_spring=1.5,
        seed=42
    )
    
    # Baseline: no spatial noise
    print("\n=== BASELINE (no Brownian motion) ===")
    hist_baseline = run_lattice_sim(
        sigma_position_noisy=0.0,
        sigma_theta_noisy=0.0,
        sigma_theta_others=0.5,
        **common_params
    )
    
    # Noisy: center oscillator undergoes Brownian motion
    print("\n=== NOISY (Brownian motion at center) ===")
    hist_noisy = run_lattice_sim(
        sigma_position_noisy=0.5,  # Spatial noise amplitude
        sigma_theta_noisy=0.5,      # Can also add phase noise
        sigma_theta_others=0.5,
        **common_params
    )
    
    # Compare radial displacements and neighbor distances
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))
    
    # Top left: radial displacement over time
    ax1.plot(hist_baseline['t'], hist_baseline['radial_displacement'], 
             'k--', linewidth=2.5, label='Baseline', alpha=0.8)
    ax1.plot(hist_noisy['t'], hist_noisy['radial_displacement'], 
             'r-', linewidth=2.5, label='Brownian', alpha=0.9)
    ax1.axhline(y=0, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    ax1.fill_between(hist_noisy['t'], 0, hist_noisy['radial_displacement'], 
                     where=(np.array(hist_noisy['radial_displacement']) > 0),
                     color='green', alpha=0.2)
    ax1.fill_between(hist_noisy['t'], 0, hist_noisy['radial_displacement'], 
                     where=(np.array(hist_noisy['radial_displacement']) < 0),
                     color='red', alpha=0.2)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Mean displacement toward noisy', fontsize=12)
    ax1.set_title('Radial Displacement (+ = attraction, - = repulsion)', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Top right: neighbor distances over time
    ax2.plot(hist_baseline['t'], hist_baseline['neighbor_dist'], 
             'k--', linewidth=2.5, label='Baseline', alpha=0.8)
    ax2.plot(hist_noisy['t'], hist_noisy['neighbor_dist'], 
             'r-', linewidth=2.5, label='Brownian', alpha=0.9)
    ax2.axhline(y=common_params['spacing'], color='gray', linestyle=':', 
               linewidth=2, alpha=0.5, label='Equilibrium spacing')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Mean neighbor distance', fontsize=12)
    ax2.set_title('Neighbor Distances to Noisy Oscillator', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Bottom left: histogram of radial displacements
    ax3.hist(hist_baseline['radial_displacement'][-20:], bins=15, alpha=0.6, 
            label='Baseline', color='black', edgecolor='black')
    ax3.hist(hist_noisy['radial_displacement'][-20:], bins=15, alpha=0.6, 
            label='Brownian', color='red', edgecolor='darkred')
    ax3.axvline(x=0, color='gray', linestyle='--', linewidth=2, label='No displacement')
    ax3.set_xlabel('Radial displacement', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution (final 20% of simulation)', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Bottom right: histogram of neighbor distances
    ax4.hist(hist_baseline['neighbor_dist'][-20:], bins=15, alpha=0.6, 
            label='Baseline', color='black', edgecolor='black')
    ax4.hist(hist_noisy['neighbor_dist'][-20:], bins=15, alpha=0.6, 
            label='Brownian', color='red', edgecolor='darkred')
    ax4.axvline(x=common_params['spacing'], color='gray', linestyle='--', 
               linewidth=2, label='Equilibrium')
    ax4.set_xlabel('Neighbor distance', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution (final 20% of simulation)', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show final states
    print("\nBaseline final state:")
    visualize_emergence(hist_baseline, frame=-1)
    
    print("\nNoisy final state (with Brownian motion):")
    visualize_emergence(hist_noisy, frame=-1)
    
    # Print statistics
    print("\n=== STATISTICS ===")
    print(f"Baseline mean radial displacement: {np.mean(hist_baseline['radial_displacement'][-20:]):.6f}")
    print(f"Brownian mean radial displacement: {np.mean(hist_noisy['radial_displacement'][-20:]):.6f}")
    print(f"Net displacement toward noisy: {np.mean(hist_noisy['radial_displacement'][-20:]) - np.mean(hist_baseline['radial_displacement'][-20:]):.6f}")
    print(f"\nBaseline mean neighbor distance: {np.mean(hist_baseline['neighbor_dist'][-20:]):.4f}")
    print(f"Brownian mean neighbor distance: {np.mean(hist_noisy['neighbor_dist'][-20:]):.4f}")
    print(f"Equilibrium spacing: {common_params['spacing']:.4f}")
    print(f"Distance shift: {np.mean(hist_noisy['neighbor_dist'][-20:]) - np.mean(hist_baseline['neighbor_dist'][-20:]):.4f}")

# -----------------------------
# Demo
# -----------------------------
if __name__ == "__main__":
    compare_baseline_vs_noisy()