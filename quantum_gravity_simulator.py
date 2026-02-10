"""
Quantum Gravity Simulator

A package to simulate quantum gravity using mass-spring networks.
Supports 1D and 2D (square/triangular) configurations with Brownian motion.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from tqdm import tqdm


class MassSpringNetwork:
    """Base class for mass-spring network simulation."""
    
    def __init__(self, mass: float = 1.0, spring_constant: float = 1.0, 
                 damping: float = 0.1, dt: float = 0.01, temperature: float = 1.0,
                 rest_length: float = 1.0):
        """
        Initialize the mass-spring network.
        
        Args:
            mass: Mass of each node
            spring_constant: Spring constant for connections
            damping: Damping coefficient
            dt: Time step for simulation
            temperature: Temperature for Brownian motion
            rest_length: Equilibrium length of springs (default: 1.0)
        """
        self.mass = mass
        self.k = spring_constant
        self.damping = damping
        self.dt = dt
        self.temperature = temperature
        self.rest_length = rest_length
        
        self.positions = None
        self.velocities = None
        self.connections = []
        self.center_idx = None
        self.displacement_history = []
        self.initial_positions = None  # Store initial positions
        self.radial_displacement_history = []  # Track migration toward center
        self.neighbor_distance_history = []  # Track neighbor distances over time
        
    def compute_forces(self) -> np.ndarray:
        """
        Compute forces on all masses from spring connections.
        
        For 2D networks with periodic boundaries, uses periodic_vector() to calculate
        the minimum displacement considering wrapping. For 1D networks or 2D networks
        without periodic boundaries, uses direct position differences.
        
        Springs now use Hooke's law with equilibrium rest length:
        F = -k * (distance - rest_length)
        """
        forces = np.zeros_like(self.positions)
        
        # Check if this network has periodic boundary support
        has_periodic = hasattr(self, 'periodic_vector')
        
        # Spring forces with equilibrium position
        for i, j in self.connections:
            if has_periodic:
                # Use periodic boundary conditions for distance calculation
                displacement = self.periodic_vector(self.positions[i], self.positions[j])
            else:
                # Direct displacement without periodic boundaries
                displacement = self.positions[j] - self.positions[i]
            
            distance = np.linalg.norm(displacement)
            if distance > 0:
                # Hooke's law with rest length: F = -k(r - r0)
                force_magnitude = self.k * (distance - self.rest_length)
                force_direction = displacement / distance
                force = force_magnitude * force_direction
                forces[i] += force
                forces[j] -= force
        
        # Damping forces
        forces -= self.damping * self.velocities
        
        return forces
    
    def apply_brownian_motion(self):
        """Apply Brownian random walk to the center node."""
        if self.center_idx is not None:
            # Brownian motion: random displacement proportional to sqrt(temperature)
            ndim = self.positions.shape[1]
            random_force = np.random.randn(ndim) * np.sqrt(self.temperature)
            self.velocities[self.center_idx] += random_force * self.dt / self.mass
    
    def step(self):
        """Perform one simulation step."""
        # Compute forces
        forces = self.compute_forces()
        
        # Update velocities (except for center node which gets Brownian motion)
        accelerations = forces / self.mass
        self.velocities += accelerations * self.dt
        
        # Apply Brownian motion to center node
        self.apply_brownian_motion()
        
        # Update positions
        self.positions += self.velocities * self.dt
        
        # Track displacements of non-central masses
        self.track_displacements()
    
    def track_displacements(self):
        """
        Track displacements of all masses from their initial positions.
        Also tracks radial displacement toward center and neighbor distances.
        
        Note: This tracks all masses including the center node. Use get_displacements()
        to retrieve the final displacements.
        """
        if len(self.displacement_history) == 0:
            # Initialize with current positions as reference
            self.displacement_history.append(self.positions.copy())
        else:
            # Calculate displacement from initial position
            displacements = self.positions - self.displacement_history[0]
            self.displacement_history.append(displacements.copy())
        
        # Track radial displacement (migration toward center)
        if self.center_idx is not None and self.initial_positions is not None:
            radial_disp = self.compute_radial_displacement()
            self.radial_displacement_history.append(radial_disp)
            
            # Track neighbor distances
            neighbor_dist = self.compute_neighbor_distances()
            self.neighbor_distance_history.append(neighbor_dist)
    
    def simulate(self, steps: int, show_progress: bool = True):
        """
        Run the simulation for a given number of steps.
        
        Args:
            steps: Number of simulation steps
            show_progress: Whether to show a progress bar
        """
        self.displacement_history = []
        self.radial_displacement_history = []
        self.neighbor_distance_history = []
        # Store initial positions before simulation
        if self.initial_positions is None:
            self.initial_positions = self.positions.copy()
        
        iterator = tqdm(range(steps), desc="Simulating") if show_progress else range(steps)
        for _ in iterator:
            self.step()
    
    def get_initial_positions(self) -> np.ndarray:
        """Get initial positions of all masses."""
        if self.initial_positions is not None:
            return self.initial_positions.copy()
        return self.positions.copy()
    
    def get_final_positions(self) -> np.ndarray:
        """Get final positions of all masses."""
        return self.positions.copy()
    
    def get_displacements(self) -> np.ndarray:
        """Get displacements of all non-central masses."""
        if len(self.displacement_history) > 1:
            return self.displacement_history[-1]
        return np.zeros_like(self.positions)
    
    def get_relative_displacements(self) -> np.ndarray:
        """
        Get relative displacements of all masses.
        
        Relative displacement is computed by subtracting the mean displacement
        from each individual displacement. This removes bulk translation of the
        system and reveals the oscillation patterns relative to the moving grid,
        rather than showing absolute motion in the lab frame.
        
        Returns:
            Array of relative displacement vectors
        """
        displacements = self.get_displacements()
        # Calculate mean displacement across all masses
        mean_displacement = np.mean(displacements, axis=0)
        # Subtract mean to get relative displacements
        relative_displacements = displacements - mean_displacement
        return relative_displacements
    
    def compute_radial_displacement(self) -> float:
        """
        Compute mean displacement of non-central masses toward/away from the center node.
        
        Positive value means masses have moved toward the center (attraction).
        Negative value means masses have moved away from the center (repulsion).
        
        Returns:
            Mean radial displacement toward center (positive = attraction)
        """
        if self.center_idx is None or self.initial_positions is None:
            return 0.0
        
        radial_disps = []
        center_pos_current = self.positions[self.center_idx]
        center_pos_initial = self.initial_positions[self.center_idx]
        
        # Check if this network has periodic boundary support
        has_periodic = hasattr(self, 'periodic_vector')
        
        for i in range(len(self.positions)):
            if i == self.center_idx:
                continue
            
            # Calculate distance from mass i to center (current and initial)
            if has_periodic:
                # Use periodic boundaries for distance calculation
                r_current = self.periodic_vector(self.positions[i], center_pos_current)
                r_initial = self.periodic_vector(self.initial_positions[i], center_pos_initial)
            else:
                # Direct distance without periodic boundaries
                r_current = center_pos_current - self.positions[i]
                r_initial = center_pos_initial - self.initial_positions[i]
            
            dist_current = np.linalg.norm(r_current)
            dist_initial = np.linalg.norm(r_initial)
            
            # Positive radial_disp means moved toward center
            radial_disp = dist_initial - dist_current
            radial_disps.append(radial_disp)
        
        return np.mean(radial_disps) if radial_disps else 0.0
    
    def compute_neighbor_distances(self) -> float:
        """
        Compute mean distance from center node to its immediate neighbors.
        
        This measures if the neighbors maintain their equilibrium spacing
        or if they're drawn closer/pushed away from the center oscillator.
        
        Returns:
            Mean distance to center node's neighbors
        """
        if self.center_idx is None:
            return self.rest_length
        
        # Find neighbors of center node (connected via springs)
        neighbor_indices = []
        for i, j in self.connections:
            if i == self.center_idx:
                neighbor_indices.append(j)
            elif j == self.center_idx:
                neighbor_indices.append(i)
        
        if not neighbor_indices:
            return self.rest_length
        
        # Check if this network has periodic boundary support
        has_periodic = hasattr(self, 'periodic_vector')
        
        # Compute distances
        distances = []
        center_pos = self.positions[self.center_idx]
        for neighbor_idx in neighbor_indices:
            if has_periodic:
                r_vec = self.periodic_vector(self.positions[neighbor_idx], center_pos)
            else:
                r_vec = center_pos - self.positions[neighbor_idx]
            dist = np.linalg.norm(r_vec)
            distances.append(dist)
        
        return np.mean(distances) if distances else self.rest_length
    
    def get_radial_displacement_history(self) -> np.ndarray:
        """
        Get history of radial displacements over time.
        
        Returns:
            Array of mean radial displacements at each tracked time step
        """
        return np.array(self.radial_displacement_history) if self.radial_displacement_history else np.array([])
    
    def get_neighbor_distance_history(self) -> np.ndarray:
        """
        Get history of neighbor distances over time.
        
        Returns:
            Array of mean neighbor distances at each tracked time step
        """
        return np.array(self.neighbor_distance_history) if self.neighbor_distance_history else np.array([])


class Network1D(MassSpringNetwork):
    """1D mass-spring network."""
    
    def __init__(self, n_masses: int, **kwargs):
        """
        Initialize 1D network.
        
        Args:
            n_masses: Number of masses in the chain
            **kwargs: Additional parameters for MassSpringNetwork
        """
        super().__init__(**kwargs)
        self.n_masses = n_masses
        self._initialize_network()
    
    def _initialize_network(self):
        """Initialize the 1D network structure."""
        # Create masses along a line
        self.positions = np.zeros((self.n_masses, 1))
        for i in range(self.n_masses):
            self.positions[i, 0] = i * 1.0
        
        self.velocities = np.zeros_like(self.positions)
        
        # Connect adjacent masses
        self.connections = []
        for i in range(self.n_masses - 1):
            self.connections.append((i, i + 1))
        
        # Center node is the middle one
        self.center_idx = self.n_masses // 2
        
        # Store initial positions
        self.initial_positions = self.positions.copy()


class Network2DSquare(MassSpringNetwork):
    """2D mass-spring network with square lattice and periodic boundary conditions."""
    
    def __init__(self, size: int, **kwargs):
        """
        Initialize 2D square lattice network with periodic boundary conditions.
        
        Args:
            size: Size of the square lattice (size x size)
            **kwargs: Additional parameters for MassSpringNetwork
        """
        super().__init__(**kwargs)
        self.size = size
        self.n_masses = size * size
        self.spacing = 1.0  # Grid spacing for periodic boundaries
        self._initialize_network()
    
    def _initialize_network(self):
        """Initialize the 2D square lattice structure with periodic boundaries."""
        # Create masses in a grid
        self.positions = np.zeros((self.n_masses, 2))
        self.idx_map = {}
        
        idx = 0
        for i in range(self.size):
            for j in range(self.size):
                self.positions[idx] = [i * self.spacing, j * self.spacing]
                self.idx_map[(i, j)] = idx
                idx += 1
        
        self.velocities = np.zeros_like(self.positions)
        
        # Connect adjacent masses with periodic boundaries (4 neighbors)
        self.connections = []
        for i in range(self.size):
            for j in range(self.size):
                idx1 = self.idx_map[(i, j)]
                # Right neighbor (with wrapping)
                idx2 = self.idx_map[(i, (j + 1) % self.size)]
                self.connections.append((idx1, idx2))
                # Bottom neighbor (with wrapping)
                idx2 = self.idx_map[((i + 1) % self.size, j)]
                self.connections.append((idx1, idx2))
        
        # Center node
        center_i, center_j = self.size // 2, self.size // 2
        self.center_idx = self.idx_map[(center_i, center_j)]
        
        # Store initial positions
        self.initial_positions = self.positions.copy()
    
    def periodic_vector(self, source_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """
        Get minimum vector from source_pos to target_pos considering periodic boundaries.
        
        Args:
            source_pos: Position of source node
            target_pos: Position of target node
            
        Returns:
            Minimum displacement vector from source to target considering periodic wrapping
        """
        dx = target_pos - source_pos
        # Apply periodic boundary conditions
        Lx = self.size * self.spacing
        Ly = self.size * self.spacing
        dx[0] = dx[0] - Lx * np.round(dx[0] / Lx)
        dx[1] = dx[1] - Ly * np.round(dx[1] / Ly)
        return dx


class Network2DTriangular(MassSpringNetwork):
    """2D mass-spring network with triangular lattice and periodic boundary conditions."""
    
    def __init__(self, size: int, **kwargs):
        """
        Initialize 2D triangular lattice network with periodic boundary conditions.
        
        Args:
            size: Size of the triangular lattice
            **kwargs: Additional parameters for MassSpringNetwork
        """
        super().__init__(**kwargs)
        self.size = size
        self.n_masses = size * size
        self.spacing = 1.0  # Grid spacing for periodic boundaries
        self._initialize_network()
    
    def _initialize_network(self):
        """Initialize the 2D triangular lattice structure with periodic boundaries."""
        # Create masses in a triangular grid
        self.positions = np.zeros((self.n_masses, 2))
        self.idx_map = {}
        
        idx = 0
        for i in range(self.size):
            for j in range(self.size):
                # Offset every other row for triangular lattice
                x = i * self.spacing
                y = j * self.spacing + (0.5 if i % 2 == 1 else 0.0)
                self.positions[idx] = [x, y]
                self.idx_map[(i, j)] = idx
                idx += 1
        
        self.velocities = np.zeros_like(self.positions)
        
        # Connect adjacent masses with periodic boundaries (6 neighbors for triangular lattice)
        self.connections = []
        for i in range(self.size):
            for j in range(self.size):
                idx1 = self.idx_map[(i, j)]
                
                # Right neighbor (with wrapping)
                idx2 = self.idx_map[(i, (j + 1) % self.size)]
                self.connections.append((idx1, idx2))
                
                # Bottom neighbor (with wrapping)
                idx2 = self.idx_map[((i + 1) % self.size, j)]
                self.connections.append((idx1, idx2))
                
                # Diagonal connections for triangular lattice (with wrapping)
                if i % 2 == 0:
                    # Even rows: connect to bottom-left (with wrapping)
                    idx2 = self.idx_map[((i + 1) % self.size, (j - 1) % self.size)]
                    self.connections.append((idx1, idx2))
                else:
                    # Odd rows: connect to bottom-right (with wrapping)
                    idx2 = self.idx_map[((i + 1) % self.size, (j + 1) % self.size)]
                    self.connections.append((idx1, idx2))
        
        # Center node
        center_i, center_j = self.size // 2, self.size // 2
        self.center_idx = self.idx_map[(center_i, center_j)]
        
        # Store initial positions
        self.initial_positions = self.positions.copy()
    
    def periodic_vector(self, source_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """
        Get minimum vector from source_pos to target_pos considering periodic boundaries.
        
        Args:
            source_pos: Position of source node
            target_pos: Position of target node
            
        Returns:
            Minimum displacement vector from source to target considering periodic wrapping
        """
        dx = target_pos - source_pos
        # Apply periodic boundary conditions
        # For triangular lattice, we still use rectangular periodic boundaries
        Lx = self.size * self.spacing
        # For triangular lattice, effective y-size accounting for offset
        Ly = self.size * self.spacing
        dx[0] = dx[0] - Lx * np.round(dx[0] / Lx)
        dx[1] = dx[1] - Ly * np.round(dx[1] / Ly)
        return dx


def visualize_network(network: MassSpringNetwork, title: str = "Mass-Spring Network", 
                     amplification_factor: float = 5.0):
    """
    Visualize the final state of the network with relative displacement vectors.
    Shows both initial and displaced positions with motion relative to the grid.
    
    Args:
        network: The network to visualize
        title: Title for the plot
        amplification_factor: Factor to amplify displacement vectors for visibility (default: 5.0)
    """
    positions = network.get_final_positions()
    displacements = network.get_displacements()
    relative_displacements = network.get_relative_displacements()
    initial_positions = network.get_initial_positions()
    
    if positions.shape[1] == 1:
        # 1D visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot initial and final positions with relative displacement vectors
        ax1.scatter(initial_positions[:, 0], np.zeros(len(initial_positions)), 
                   c='lightblue', s=100, alpha=0.5, label='Initial positions', marker='o')
        ax1.scatter(positions[:, 0], np.zeros(len(positions)), 
                   c='blue', s=100, label='Final positions', marker='o')
        
        # Draw relative displacement vectors (amplified and thicker)
        for i in range(len(positions)):
            if i != network.center_idx:  # Skip center node for clarity
                # Use relative displacement instead of absolute
                rel_disp = relative_displacements[i, 0] * amplification_factor
                ax1.arrow(initial_positions[i, 0], 0, 
                         rel_disp, 0,
                         head_width=0.15, head_length=0.08, fc='green', ec='green', 
                         alpha=0.7, width=0.04)  # Thicker arrows
        
        if network.center_idx is not None:
            ax1.scatter(positions[network.center_idx, 0], 0, 
                       c='red', s=200, marker='*', label='Center node', zorder=5)
        ax1.set_xlabel('Position')
        ax1.set_title(f'{title} - Relative Motion Vectors (amplified {amplification_factor}x)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot relative displacements
        non_center_mask = np.ones(len(positions), dtype=bool)
        if network.center_idx is not None:
            non_center_mask[network.center_idx] = False
        
        indices = np.arange(len(positions))
        ax2.bar(indices[non_center_mask], 
                relative_displacements[non_center_mask, 0], 
                color='green')
        ax2.set_xlabel('Mass Index')
        ax2.set_ylabel('Relative Displacement')
        ax2.set_title('Relative Displacements of Non-Central Masses')
        ax2.grid(True)
        
    else:
        # 2D visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot connections at initial positions (faint)
        for i, j in network.connections:
            pos_i, pos_j = initial_positions[i], initial_positions[j]
            ax1.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                    'gray', alpha=0.15, linewidth=0.5)
        
        # Plot connections at final positions
        for i, j in network.connections:
            pos_i, pos_j = positions[i], positions[j]
            ax1.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                    'b-', alpha=0.3, linewidth=0.5)
        
        # Plot initial positions (faint)
        ax1.scatter(initial_positions[:, 0], initial_positions[:, 1], 
                   c='lightblue', s=50, alpha=0.4, label='Initial positions')
        
        # Plot final positions
        ax1.scatter(positions[:, 0], positions[:, 1], 
                   c='blue', s=50, label='Final positions')
        
        # Draw relative displacement vectors for non-center nodes (amplified and thicker)
        for i in range(len(positions)):
            if i != network.center_idx:
                # Use relative displacement amplified for visibility
                rel_disp = relative_displacements[i] * amplification_factor
                ax1.arrow(initial_positions[i, 0], initial_positions[i, 1],
                         rel_disp[0], rel_disp[1],
                         head_width=0.08, head_length=0.05, 
                         fc='green', ec='green', alpha=0.7, width=0.02)  # Thicker arrows
        
        if network.center_idx is not None:
            ax1.scatter(positions[network.center_idx, 0], 
                       positions[network.center_idx, 1],
                       c='red', s=200, marker='*', label='Center node', zorder=5)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title(f'{title} - Relative Motion Vectors (amplified {amplification_factor}x)')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')
        
        # Plot relative displacement magnitudes
        non_center_mask = np.ones(len(positions), dtype=bool)
        if network.center_idx is not None:
            non_center_mask[network.center_idx] = False
        
        relative_displacement_magnitudes = np.linalg.norm(relative_displacements[non_center_mask], axis=1)
        scatter = ax2.scatter(positions[non_center_mask, 0], 
                            positions[non_center_mask, 1],
                            c=relative_displacement_magnitudes, 
                            s=100, cmap='viridis')
        plt.colorbar(scatter, ax=ax2, label='Relative Displacement Magnitude')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_title('Relative Displacement Magnitudes of Non-Central Masses')
        ax2.grid(True)
        ax2.axis('equal')
    
    plt.tight_layout()
    return fig


def print_results(network: MassSpringNetwork):
    """
    Print simulation results.
    
    Args:
        network: The network to report results for
    """
    positions = network.get_final_positions()
    displacements = network.get_displacements()
    
    print("\n" + "="*60)
    print("QUANTUM GRAVITY SIMULATION RESULTS")
    print("="*60)
    
    print(f"\nNetwork Configuration:")
    print(f"  - Number of masses: {len(positions)}")
    print(f"  - Spring constant: {network.k}")
    print(f"  - Temperature: {network.temperature}")
    print(f"  - Mass: {network.mass}")
    print(f"  - Rest length: {network.rest_length}")
    
    print(f"\nFinal Positions:")
    for i, pos in enumerate(positions):
        marker = " (CENTER)" if i == network.center_idx else ""
        print(f"  Mass {i}: {pos}{marker}")
    
    print(f"\nDisplacements of Non-Central Masses:")
    for i, disp in enumerate(displacements):
        if i != network.center_idx:
            mag = np.linalg.norm(disp)
            print(f"  Mass {i}: {disp} (magnitude: {mag:.4f})")
    
    # Statistics
    non_center_mask = np.ones(len(positions), dtype=bool)
    if network.center_idx is not None:
        non_center_mask[network.center_idx] = False
    
    displacement_mags = np.linalg.norm(displacements[non_center_mask], axis=1)
    print(f"\nDisplacement Statistics:")
    print(f"  - Mean: {np.mean(displacement_mags):.4f}")
    print(f"  - Std Dev: {np.std(displacement_mags):.4f}")
    print(f"  - Max: {np.max(displacement_mags):.4f}")
    print(f"  - Min: {np.min(displacement_mags):.4f}")
    
    # Radial displacement statistics
    radial_history = network.get_radial_displacement_history()
    if len(radial_history) > 0:
        print(f"\nRadial Displacement (Migration toward center):")
        print(f"  - Final: {radial_history[-1]:.6f}")
        print(f"  - Mean (last 20%): {np.mean(radial_history[-max(1, len(radial_history)//5):]):.6f}")
        print(f"  - (Positive = attraction, Negative = repulsion)")
    
    # Neighbor distance statistics
    neighbor_history = network.get_neighbor_distance_history()
    if len(neighbor_history) > 0:
        print(f"\nNeighbor Distances to Center:")
        print(f"  - Final: {neighbor_history[-1]:.4f}")
        print(f"  - Mean (last 20%): {np.mean(neighbor_history[-max(1, len(neighbor_history)//5):]):.4f}")
        print(f"  - Rest length: {network.rest_length:.4f}")
    
    print("="*60 + "\n")


def visualize_migration(network: MassSpringNetwork, title: str = "Migration Analysis"):
    """
    Visualize the migration behavior showing radial displacement and neighbor distances over time.
    
    Args:
        network: The network to visualize
        title: Title for the plot
    """
    radial_history = network.get_radial_displacement_history()
    neighbor_history = network.get_neighbor_distance_history()
    
    if len(radial_history) == 0 or len(neighbor_history) == 0:
        print("Warning: No history data to visualize. Run a simulation first.")
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Convert to time steps
    time_steps = np.arange(len(radial_history))
    
    # Plot radial displacement over time
    ax1.plot(time_steps, radial_history, 'b-', linewidth=2, label='Radial displacement')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2, label='No displacement')
    ax1.fill_between(time_steps, 0, radial_history,
                     where=(np.array(radial_history) > 0),
                     color='green', alpha=0.3, label='Attraction')
    ax1.fill_between(time_steps, 0, radial_history,
                     where=(np.array(radial_history) < 0),
                     color='red', alpha=0.3, label='Repulsion')
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Mean Radial Displacement', fontsize=12)
    ax1.set_title(f'{title} - Migration Toward/Away from Center', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot neighbor distances over time
    ax2.plot(time_steps, neighbor_history, 'r-', linewidth=2, label='Mean neighbor distance')
    ax2.axhline(y=network.rest_length, color='k', linestyle='--',
               linewidth=2, alpha=0.5, label=f'Equilibrium ({network.rest_length:.2f})')
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Distance to Center', fontsize=12)
    ax2.set_title('Neighbor Distances to Center Oscillator', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
