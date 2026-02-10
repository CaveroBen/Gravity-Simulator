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
                 damping: float = 0.1, dt: float = 0.01, temperature: float = 1.0):
        """
        Initialize the mass-spring network.
        
        Args:
            mass: Mass of each node
            spring_constant: Spring constant for connections
            damping: Damping coefficient
            dt: Time step for simulation
            temperature: Temperature for Brownian motion
        """
        self.mass = mass
        self.k = spring_constant
        self.damping = damping
        self.dt = dt
        self.temperature = temperature
        
        self.positions = None
        self.velocities = None
        self.connections = []
        self.center_idx = None
        self.displacement_history = []
        self.initial_positions = None  # Store initial positions
        
    def compute_forces(self) -> np.ndarray:
        """
        Compute forces on all masses from spring connections.
        
        For 2D networks with periodic boundaries, uses periodic_vector() to calculate
        the minimum displacement considering wrapping. For 1D networks or 2D networks
        without periodic boundaries, uses direct position differences.
        """
        forces = np.zeros_like(self.positions)
        
        # Check if this network has periodic boundary support
        has_periodic = hasattr(self, 'periodic_vector')
        
        # Spring forces
        for i, j in self.connections:
            if has_periodic:
                # Use periodic boundary conditions for distance calculation
                displacement = self.periodic_vector(self.positions[i], self.positions[j])
            else:
                # Direct displacement without periodic boundaries
                displacement = self.positions[j] - self.positions[i]
            
            distance = np.linalg.norm(displacement)
            if distance > 0:
                force_magnitude = self.k * distance
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
    
    def simulate(self, steps: int, show_progress: bool = True):
        """
        Run the simulation for a given number of steps.
        
        Args:
            steps: Number of simulation steps
            show_progress: Whether to show a progress bar
        """
        self.displacement_history = []
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


def visualize_network(network: MassSpringNetwork, title: str = "Mass-Spring Network"):
    """
    Visualize the final state of the network with displacement vectors.
    Shows both initial and displaced positions.
    
    Args:
        network: The network to visualize
        title: Title for the plot
    """
    positions = network.get_final_positions()
    displacements = network.get_displacements()
    initial_positions = network.get_initial_positions()
    
    if positions.shape[1] == 1:
        # 1D visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot initial and final positions with displacement vectors
        ax1.scatter(initial_positions[:, 0], np.zeros(len(initial_positions)), 
                   c='lightblue', s=100, alpha=0.5, label='Initial positions', marker='o')
        ax1.scatter(positions[:, 0], np.zeros(len(positions)), 
                   c='blue', s=100, label='Final positions', marker='o')
        
        # Draw displacement vectors
        for i in range(len(positions)):
            if i != network.center_idx:  # Skip center node for clarity
                ax1.arrow(initial_positions[i, 0], 0, 
                         displacements[i, 0], 0,
                         head_width=0.1, head_length=0.05, fc='green', ec='green', 
                         alpha=0.7, width=0.02)
        
        if network.center_idx is not None:
            ax1.scatter(positions[network.center_idx, 0], 0, 
                       c='red', s=200, marker='*', label='Center node', zorder=5)
        ax1.set_xlabel('Position')
        ax1.set_title(f'{title} - Positions and Displacement Vectors')
        ax1.legend()
        ax1.grid(True)
        
        # Plot displacements
        non_center_mask = np.ones(len(positions), dtype=bool)
        if network.center_idx is not None:
            non_center_mask[network.center_idx] = False
        
        indices = np.arange(len(positions))
        ax2.bar(indices[non_center_mask], 
                displacements[non_center_mask, 0], 
                color='green')
        ax2.set_xlabel('Mass Index')
        ax2.set_ylabel('Displacement')
        ax2.set_title('Displacements of Non-Central Masses')
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
        
        # Draw displacement vectors for non-center nodes (center node vector omitted for clarity)
        for i in range(len(positions)):
            if i != network.center_idx:
                ax1.arrow(initial_positions[i, 0], initial_positions[i, 1],
                         displacements[i, 0], displacements[i, 1],
                         head_width=0.05, head_length=0.03, 
                         fc='green', ec='green', alpha=0.6, width=0.01)
        
        if network.center_idx is not None:
            ax1.scatter(positions[network.center_idx, 0], 
                       positions[network.center_idx, 1],
                       c='red', s=200, marker='*', label='Center node', zorder=5)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title(f'{title} - Positions and Displacement Vectors')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')
        
        # Plot displacement magnitudes
        non_center_mask = np.ones(len(positions), dtype=bool)
        if network.center_idx is not None:
            non_center_mask[network.center_idx] = False
        
        displacement_magnitudes = np.linalg.norm(displacements[non_center_mask], axis=1)
        scatter = ax2.scatter(positions[non_center_mask, 0], 
                            positions[non_center_mask, 1],
                            c=displacement_magnitudes, 
                            s=100, cmap='viridis')
        plt.colorbar(scatter, ax=ax2, label='Displacement Magnitude')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_title('Displacement Magnitudes of Non-Central Masses')
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
    
    print("="*60 + "\n")
