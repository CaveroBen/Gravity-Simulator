"""
Quantum Gravity Simulator

A package to simulate quantum gravity using mass-spring networks.
Supports 1D and 2D (square/triangular) configurations with Brownian motion.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
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
        HISTORY_TAIL_FRACTION = 0.2  # Analyze last 20% of simulation
        tail_start = -max(1, int(len(radial_history) * HISTORY_TAIL_FRACTION))
        print(f"\nRadial Displacement (Migration toward center):")
        print(f"  - Final: {radial_history[-1]:.6f}")
        print(f"  - Mean (last 20%): {np.mean(radial_history[tail_start:]):.6f}")
        print(f"  - (Positive = attraction, Negative = repulsion)")
    
    # Neighbor distance statistics
    neighbor_history = network.get_neighbor_distance_history()
    if len(neighbor_history) > 0:
        tail_start = -max(1, int(len(neighbor_history) * HISTORY_TAIL_FRACTION))
        print(f"\nNeighbor Distances to Center:")
        print(f"  - Final: {neighbor_history[-1]:.4f}")
        print(f"  - Mean (last 20%): {np.mean(neighbor_history[tail_start:]):.4f}")
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


def run_multiple_simulations(network_class, network_params: dict, 
                            simulation_steps: int, n_simulations: int = 10,
                            show_progress: bool = True) -> dict:
    """
    Run multiple simulations and collect results for averaging.
    
    Args:
        network_class: The network class to use (Network1D, Network2DSquare, or Network2DTriangular)
        network_params: Dictionary of parameters to pass to the network constructor
        simulation_steps: Number of steps to run each simulation
        n_simulations: Number of simulations to run (default: 10)
        show_progress: Whether to show progress bars
    
    Returns:
        Dictionary containing:
            - 'initial_positions': Initial positions (same for all runs)
            - 'average_final_positions': Average final positions across all simulations
            - 'average_displacements': Average displacements across all simulations
            - 'std_displacements': Standard deviation of displacements
            - 'all_final_positions': List of final positions from each simulation
            - 'all_displacements': List of displacements from each simulation
            - 'n_simulations': Number of simulations run
    """
    all_final_positions = []
    all_displacements = []
    initial_positions = None
    
    print(f"\nRunning {n_simulations} simulations...")
    
    for i in tqdm(range(n_simulations), desc="Simulations", disable=not show_progress):
        # Create a fresh network for each simulation
        network = network_class(**network_params)
        
        # Store initial positions from the first simulation
        if initial_positions is None:
            initial_positions = network.get_initial_positions()
        
        # Run the simulation
        network.simulate(steps=simulation_steps, show_progress=False)
        
        # Collect results
        final_positions = network.get_final_positions()
        displacements = network.get_displacements()
        
        all_final_positions.append(final_positions)
        all_displacements.append(displacements)
    
    # Calculate averages
    average_final_positions = np.mean(all_final_positions, axis=0)
    average_displacements = np.mean(all_displacements, axis=0)
    std_displacements = np.std(all_displacements, axis=0)
    
    return {
        'initial_positions': initial_positions,
        'average_final_positions': average_final_positions,
        'average_displacements': average_displacements,
        'std_displacements': std_displacements,
        'all_final_positions': all_final_positions,
        'all_displacements': all_displacements,
        'n_simulations': n_simulations
    }


def visualize_averaged_results(results: dict, network_class, network_params: dict, 
                              title: str = "Averaged Simulation Results",
                              amplification_factor: float = 5.0):
    """
    Visualize averaged results from multiple simulations.
    
    Args:
        results: Dictionary returned by run_multiple_simulations
        network_class: The network class used (Network1D, Network2DSquare, or Network2DTriangular)
        network_params: Dictionary of network parameters
        title: Title for the plot
        amplification_factor: Factor to amplify displacement vectors for visibility
    
    Returns:
        The matplotlib figure
    """
    initial_positions = results['initial_positions']
    average_final_positions = results['average_final_positions']
    average_displacements = results['average_displacements']
    n_simulations = results['n_simulations']
    
    # Determine network type from initial positions shape
    is_1d = initial_positions.shape[1] == 1
    
    # Create a temporary network to get center_idx and connections
    temp_network = network_class(**network_params)
    center_idx = temp_network.center_idx
    connections = temp_network.connections
    
    if is_1d:
        # 1D visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot initial and averaged final positions with displacement vectors
        ax1.scatter(initial_positions[:, 0], np.zeros(len(initial_positions)), 
                   c='lightblue', s=100, alpha=0.5, label='Initial positions', marker='o')
        ax1.scatter(average_final_positions[:, 0], np.zeros(len(average_final_positions)), 
                   c='blue', s=100, label=f'Average final positions (n={n_simulations})', marker='o')
        
        # Draw average displacement vectors (amplified and thicker)
        for i in range(len(average_final_positions)):
            if i != center_idx:  # Skip center node for clarity
                avg_disp = average_displacements[i, 0] * amplification_factor
                ax1.arrow(initial_positions[i, 0], 0, 
                         avg_disp, 0,
                         head_width=0.15, head_length=0.08, fc='green', ec='green', 
                         alpha=0.7, width=0.04)
        
        if center_idx is not None:
            ax1.scatter(average_final_positions[center_idx, 0], 0, 
                       c='red', s=200, marker='*', label='Center node', zorder=5)
        ax1.set_xlabel('Position')
        ax1.set_title(f'{title} - Average Displacement Vectors (amplified {amplification_factor}x, n={n_simulations})')
        ax1.legend()
        ax1.grid(True)
        
        # Plot average displacements
        non_center_mask = np.ones(len(average_final_positions), dtype=bool)
        if center_idx is not None:
            non_center_mask[center_idx] = False
        
        indices = np.arange(len(average_final_positions))
        ax2.bar(indices[non_center_mask], 
                average_displacements[non_center_mask, 0], 
                color='green')
        ax2.set_xlabel('Mass Index')
        ax2.set_ylabel('Average Displacement')
        ax2.set_title(f'Average Displacements of Non-Central Masses (n={n_simulations})')
        ax2.grid(True)
        
    else:
        # 2D visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot connections at initial positions (faint)
        for i, j in connections:
            pos_i, pos_j = initial_positions[i], initial_positions[j]
            ax1.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                    'gray', alpha=0.15, linewidth=0.5)
        
        # Plot connections at average final positions
        for i, j in connections:
            pos_i, pos_j = average_final_positions[i], average_final_positions[j]
            ax1.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                    'b-', alpha=0.3, linewidth=0.5)
        
        # Plot initial positions (faint)
        ax1.scatter(initial_positions[:, 0], initial_positions[:, 1], 
                   c='lightblue', s=50, alpha=0.4, label='Initial positions')
        
        # Plot average final positions
        ax1.scatter(average_final_positions[:, 0], average_final_positions[:, 1], 
                   c='blue', s=50, label=f'Average final positions (n={n_simulations})')
        
        # Draw average displacement vectors for non-center nodes (amplified and thicker)
        for i in range(len(average_final_positions)):
            if i != center_idx:
                avg_disp = average_displacements[i] * amplification_factor
                ax1.arrow(initial_positions[i, 0], initial_positions[i, 1],
                         avg_disp[0], avg_disp[1],
                         head_width=0.08, head_length=0.05, 
                         fc='green', ec='green', alpha=0.7, width=0.02)
        
        if center_idx is not None:
            ax1.scatter(average_final_positions[center_idx, 0], 
                       average_final_positions[center_idx, 1],
                       c='red', s=200, marker='*', label='Center node', zorder=5)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title(f'{title} - Average Displacement Vectors (amplified {amplification_factor}x, n={n_simulations})')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')
        
        # Plot average displacement magnitudes
        non_center_mask = np.ones(len(average_final_positions), dtype=bool)
        if center_idx is not None:
            non_center_mask[center_idx] = False
        
        avg_displacement_magnitudes = np.linalg.norm(average_displacements[non_center_mask], axis=1)
        scatter = ax2.scatter(average_final_positions[non_center_mask, 0], 
                            average_final_positions[non_center_mask, 1],
                            c=avg_displacement_magnitudes, 
                            s=100, cmap='viridis')
        plt.colorbar(scatter, ax=ax2, label='Average Displacement Magnitude')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_title(f'Average Displacement Magnitudes (n={n_simulations})')
        ax2.grid(True)
        ax2.axis('equal')
    
    plt.tight_layout()
    return fig


def visualize_network_3d(network: MassSpringNetwork, title: str = "3D Displacement Visualization"):
    """
    Visualize the network in 3D where the third dimension is the magnitude of 
    displacement towards the center node.
    
    For 2D networks, this creates a 3D plot where:
    - X and Y are the spatial positions of the particles
    - Z is the magnitude of displacement towards the center (the particle with noise)
    
    Args:
        network: The network to visualize. Must be a MassSpringNetwork with 
                get_final_positions(), get_initial_positions() methods, and a 
                center_idx attribute.
        title: Title for the plot
    
    Returns:
        The matplotlib figure object, or None if the network is not 2D or 
        center_idx is not defined
    """
    positions = network.get_final_positions()
    initial_positions = network.get_initial_positions()
    
    # Check if network is 2D
    if positions.shape[1] != 2:
        print("Warning: 3D visualization is only supported for 2D networks.")
        return None
    
    if network.center_idx is None:
        print("Warning: No center node defined in the network.")
        return None
    
    # Calculate displacement magnitudes toward the center
    center_pos_current = positions[network.center_idx]
    center_pos_initial = initial_positions[network.center_idx]
    
    # Check if this network has periodic boundary support
    has_periodic = hasattr(network, 'periodic_vector') and callable(getattr(network, 'periodic_vector', None))
    
    displacement_magnitudes = np.zeros(len(positions))
    
    for i in range(len(positions)):
        if i == network.center_idx:
            # The center node itself has zero "displacement toward center"
            displacement_magnitudes[i] = 0.0
        else:
            # Calculate distance from mass i to center (current and initial)
            if has_periodic:
                # Use periodic boundaries for distance calculation
                r_current = network.periodic_vector(positions[i], center_pos_current)
                r_initial = network.periodic_vector(initial_positions[i], center_pos_initial)
            else:
                # Direct distance without periodic boundaries
                r_current = center_pos_current - positions[i]
                r_initial = center_pos_initial - initial_positions[i]
            
            dist_current = np.linalg.norm(r_current)
            dist_initial = np.linalg.norm(r_initial)
            
            # Positive value means moved toward center (magnitude of displacement)
            displacement_magnitudes[i] = dist_initial - dist_current
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Separate center and non-center nodes for visualization
    non_center_mask = np.ones(len(positions), dtype=bool)
    non_center_mask[network.center_idx] = False
    
    # Plot non-center nodes
    scatter = ax.scatter(
        positions[non_center_mask, 0],
        positions[non_center_mask, 1],
        displacement_magnitudes[non_center_mask],
        c=displacement_magnitudes[non_center_mask],
        cmap='viridis',
        s=100,
        alpha=0.8,
        label='Particles'
    )
    
    # Plot center node
    ax.scatter(
        positions[network.center_idx, 0],
        positions[network.center_idx, 1],
        displacement_magnitudes[network.center_idx],
        c='red',
        s=300,
        marker='*',
        label='Center node (with noise)',
        edgecolors='darkred',
        linewidths=2
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Displacement Magnitude Toward Center', rotation=270, labelpad=20)
    
    # Labels and title
    ax.set_xlabel('X Position', fontsize=11)
    ax.set_ylabel('Y Position', fontsize=11)
    ax.set_zlabel('Displacement Toward Center', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    # Add grid for better depth perception
    ax.grid(True, alpha=0.3)
    
    # Improve viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig


def visualize_network_3d_surface(network: MassSpringNetwork, 
                                  title: str = "3D Displacement Surface Visualization",
                                  interactive: bool = True):
    """
    Visualize the network in 3D with a surface plot where the third dimension is the magnitude of 
    displacement towards the center node. The surface connects points smoothly.
    
    For 2D networks, this creates a 3D surface plot where:
    - X and Y are the spatial positions of the particles
    - Z is the magnitude of displacement towards the center (the particle with noise)
    - Points are connected by a triangulated surface mesh
    
    Args:
        network: The network to visualize. Must be a MassSpringNetwork with 
                get_final_positions(), get_initial_positions() methods, and a 
                center_idx attribute.
        title: Title for the plot
        interactive: If True, enables interactive rotation and repositioning (default: True)
    
    Returns:
        The matplotlib figure object, or None if the network is not 2D or 
        center_idx is not defined
    """
    positions = network.get_final_positions()
    initial_positions = network.get_initial_positions()
    
    # Check if network is 2D
    if positions.shape[1] != 2:
        print("Warning: 3D surface visualization is only supported for 2D networks.")
        return None
    
    if network.center_idx is None:
        print("Warning: No center node defined in the network.")
        return None
    
    # Calculate displacement magnitudes toward the center
    center_pos_current = positions[network.center_idx]
    center_pos_initial = initial_positions[network.center_idx]
    
    # Check if this network has periodic boundary support
    has_periodic = hasattr(network, 'periodic_vector') and callable(getattr(network, 'periodic_vector', None))
    
    displacement_magnitudes = np.zeros(len(positions))
    
    for i in range(len(positions)):
        if i == network.center_idx:
            # The center node itself has zero "displacement toward center"
            displacement_magnitudes[i] = 0.0
        else:
            # Calculate distance from mass i to center (current and initial)
            if has_periodic:
                # Use periodic boundaries for distance calculation
                r_current = network.periodic_vector(positions[i], center_pos_current)
                r_initial = network.periodic_vector(initial_positions[i], center_pos_initial)
            else:
                # Direct distance without periodic boundaries
                r_current = center_pos_current - positions[i]
                r_initial = center_pos_initial - initial_positions[i]
            
            dist_current = np.linalg.norm(r_current)
            dist_initial = np.linalg.norm(r_initial)
            
            # Positive value means moved toward center (magnitude of displacement)
            displacement_magnitudes[i] = dist_initial - dist_current
    
    # Create triangulation for the surface
    x = positions[:, 0]
    y = positions[:, 1]
    z = displacement_magnitudes
    
    # Create a Delaunay triangulation
    triang = Triangulation(x, y)
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_trisurf(triang, z, cmap='viridis', alpha=0.7, linewidth=0.2, 
                           edgecolor='black', antialiased=True)
    
    # Separate center and non-center nodes for visualization
    non_center_mask = np.ones(len(positions), dtype=bool)
    non_center_mask[network.center_idx] = False
    
    # Plot non-center nodes on top of surface
    ax.scatter(
        positions[non_center_mask, 0],
        positions[non_center_mask, 1],
        displacement_magnitudes[non_center_mask],
        c=displacement_magnitudes[non_center_mask],
        cmap='viridis',
        s=80,
        alpha=0.9,
        edgecolors='black',
        linewidths=1,
        label='Particles'
    )
    
    # Plot center node
    ax.scatter(
        positions[network.center_idx, 0],
        positions[network.center_idx, 1],
        displacement_magnitudes[network.center_idx],
        c='red',
        s=300,
        marker='*',
        label='Center node (with noise)',
        edgecolors='darkred',
        linewidths=2,
        zorder=10
    )
    
    # Add colorbar
    cbar = plt.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Displacement Magnitude Toward Center', rotation=270, labelpad=20)
    
    # Labels and title
    ax.set_xlabel('X Position', fontsize=11)
    ax.set_ylabel('Y Position', fontsize=11)
    ax.set_zlabel('Displacement Toward Center', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    # Add grid for better depth perception
    ax.grid(True, alpha=0.3)
    
    # Improve viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Enable interactive rotation if requested
    if interactive:
        # Enable mouse interaction for rotation
        plt.ion()  # Turn on interactive mode
        print("\n" + "="*60)
        print("INTERACTIVE 3D VISUALIZATION")
        print("="*60)
        print("You can now:")
        print("  - Click and drag to rotate the view")
        print("  - Right-click and drag to zoom")
        print("  - Use mouse wheel to zoom")
        print("  - Close the window when finished")
        print("="*60 + "\n")
    
    plt.tight_layout()
    return fig


def visualize_averaged_results_3d(results: dict, network_class, network_params: dict,
                                   title: str = "Averaged 3D Displacement Visualization"):
    """
    Visualize averaged results from multiple simulations in 3D where the z-axis
    represents average displacement magnitude.
    
    This function uses averaged data from multiple simulations instead of
    running a single simulation, providing a more robust statistical view
    of the displacement patterns.
    
    Args:
        results: Dictionary returned by run_multiple_simulations
        network_class: The network class used (Network2DSquare or Network2DTriangular)
        network_params: Dictionary of network parameters
        title: Title for the plot
    
    Returns:
        The matplotlib figure object, or None if the network is not 2D
    """
    initial_positions = results['initial_positions']
    average_final_positions = results['average_final_positions']
    average_displacements = results['average_displacements']
    n_simulations = results['n_simulations']
    
    # Check if network is 2D
    if initial_positions.shape[1] != 2:
        print("Warning: 3D visualization is only supported for 2D networks.")
        return None
    
    # Create a temporary network to get center_idx
    temp_network = network_class(**network_params)
    center_idx = temp_network.center_idx
    
    if center_idx is None:
        print("Warning: No center node defined in the network.")
        return None
    
    # Calculate average displacement magnitudes
    # These are the magnitudes of the displacement vectors from initial to final positions
    avg_displacement_magnitudes = np.linalg.norm(average_displacements, axis=1)
    
    # Create triangulation for the surface
    x = average_final_positions[:, 0]
    y = average_final_positions[:, 1]
    z = avg_displacement_magnitudes
    
    # Create a Delaunay triangulation
    triang = Triangulation(x, y)
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_trisurf(triang, z, cmap='viridis', alpha=0.7, linewidth=0,
                           antialiased=True)
    
    # Separate center and non-center nodes for visualization
    non_center_mask = np.ones(len(average_final_positions), dtype=bool)
    non_center_mask[center_idx] = False
    
    # Plot non-center nodes on top of surface
    ax.scatter(
        average_final_positions[non_center_mask, 0],
        average_final_positions[non_center_mask, 1],
        avg_displacement_magnitudes[non_center_mask],
        c=avg_displacement_magnitudes[non_center_mask],
        cmap='viridis',
        s=80,
        alpha=0.9,
        edgecolors='black',
        linewidths=1,
        label='Particles'
    )
    
    # Plot center node
    ax.scatter(
        average_final_positions[center_idx, 0],
        average_final_positions[center_idx, 1],
        avg_displacement_magnitudes[center_idx],
        c='red',
        s=300,
        marker='*',
        label='Center node (with noise)',
        edgecolors='darkred',
        linewidths=2
    )
    
    # Add colorbar
    cbar = plt.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Average Displacement Magnitude', rotation=270, labelpad=20)
    
    # Labels and title
    ax.set_xlabel('X Position', fontsize=11)
    ax.set_ylabel('Y Position', fontsize=11)
    ax.set_zlabel('Average Displacement Magnitude', fontsize=11)
    ax.set_title(f'{title}\n(Averaged over {n_simulations} simulations)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    # Add grid for better depth perception
    ax.grid(True, alpha=0.3)
    
    # Improve viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig


def animate_simulation_live(network_class, network_params: dict, simulation_steps: int,
                            update_interval: int = 50, title: str = "Live Simulation"):
    """
    Run a simulation with live animation showing the network state in real-time.
    
    This function creates a network, runs the simulation, and updates the visualization
    every 'update_interval' steps so you can watch the simulation evolve in real-time.
    
    Args:
        network_class: The network class to instantiate (e.g., Network2DSquare)
        network_params: Dictionary of parameters to pass to the network constructor
        simulation_steps: Total number of simulation steps to run
        update_interval: Number of simulation steps between visualization updates (default: 50)
        title: Title for the animation
    
    Returns:
        The final network object after simulation
    """
    # Create the network
    network = network_class(**network_params)
    
    # Check if network is 2D
    if network.positions.shape[1] != 2:
        print("Warning: Live animation is only supported for 2D networks.")
        return network
    
    # Store initial positions
    network.initial_positions = network.positions.copy()
    
    # Create figure and axis
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize data storage
    x = network.positions[:, 0]
    y = network.positions[:, 1]
    z = np.zeros(len(network.positions))
    
    # Create triangulation (this won't change during simulation)
    triang = Triangulation(x, y)
    
    # Initial plot
    surf = ax.plot_trisurf(triang, z, cmap='viridis', alpha=0.7, linewidth=0.2,
                           edgecolor='black', antialiased=True)
    
    # Scatter plot for nodes
    non_center_mask = np.ones(len(network.positions), dtype=bool)
    non_center_mask[network.center_idx] = False
    
    scatter_nodes = ax.scatter(x[non_center_mask], y[non_center_mask], z[non_center_mask],
                               c=z[non_center_mask], cmap='viridis', s=80, alpha=0.9,
                               edgecolors='black', linewidths=1)
    
    scatter_center = ax.scatter(x[network.center_idx], y[network.center_idx], z[network.center_idx],
                               c='red', s=300, marker='*', edgecolors='darkred', linewidths=2)
    
    # Add colorbar
    cbar = plt.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Displacement Magnitude Toward Center', rotation=270, labelpad=20)
    
    # Labels
    ax.set_xlabel('X Position', fontsize=11)
    ax.set_ylabel('Y Position', fontsize=11)
    ax.set_zlabel('Displacement Toward Center', fontsize=11)
    ax.set_title(f"{title} - Step 0/{simulation_steps}", fontsize=14, fontweight='bold')
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)
    
    plt.ion()
    plt.show()
    
    # Run simulation with periodic updates
    center_pos_initial = network.initial_positions[network.center_idx]
    has_periodic = hasattr(network, 'periodic_vector') and callable(getattr(network, 'periodic_vector', None))
    
    print(f"\nRunning live simulation for {simulation_steps} steps...")
    print(f"The plot will update every {update_interval} steps")
    
    for step in tqdm(range(simulation_steps)):
        # Perform one simulation step
        network.step()
        
        # Update visualization every update_interval steps
        if (step + 1) % update_interval == 0 or step == simulation_steps - 1:
            # Calculate current displacement magnitudes
            center_pos_current = network.positions[network.center_idx]
            displacement_magnitudes = np.zeros(len(network.positions))
            
            for i in range(len(network.positions)):
                if i == network.center_idx:
                    displacement_magnitudes[i] = 0.0
                else:
                    if has_periodic:
                        r_current = network.periodic_vector(network.positions[i], center_pos_current)
                        r_initial = network.periodic_vector(network.initial_positions[i], center_pos_initial)
                    else:
                        r_current = center_pos_current - network.positions[i]
                        r_initial = center_pos_initial - network.initial_positions[i]
                    
                    dist_current = np.linalg.norm(r_current)
                    dist_initial = np.linalg.norm(r_initial)
                    displacement_magnitudes[i] = dist_initial - dist_current
            
            # Update positions and z values
            x = network.positions[:, 0]
            y = network.positions[:, 1]
            z = displacement_magnitudes
            
            # Clear and redraw
            ax.clear()
            
            # Redraw surface
            triang = Triangulation(x, y)
            surf = ax.plot_trisurf(triang, z, cmap='viridis', alpha=0.7, linewidth=0.2,
                                   edgecolor='black', antialiased=True)
            
            # Redraw scatter points
            ax.scatter(x[non_center_mask], y[non_center_mask], z[non_center_mask],
                      c=z[non_center_mask], cmap='viridis', s=80, alpha=0.9,
                      edgecolors='black', linewidths=1)
            
            ax.scatter(x[network.center_idx], y[network.center_idx], z[network.center_idx],
                      c='red', s=300, marker='*', edgecolors='darkred', linewidths=2)
            
            # Update labels and title
            ax.set_xlabel('X Position', fontsize=11)
            ax.set_ylabel('Y Position', fontsize=11)
            ax.set_zlabel('Displacement Toward Center', fontsize=11)
            ax.set_title(f"{title} - Step {step+1}/{simulation_steps}", fontsize=14, fontweight='bold')
            ax.view_init(elev=20, azim=45)
            ax.grid(True, alpha=0.3)
            
            plt.draw()
            plt.pause(0.001)
    
    print("\n" + "="*60)
    print("Live simulation completed!")
    print("Close the plot window to continue...")
    print("="*60 + "\n")
    
    plt.ioff()
    return network
