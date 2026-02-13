# Quantum Gravity Simulator

A Python package to simulate quantum gravity using mass-spring networks. This simulator models spacetime as arrays (1D or 2D) of masses coupled by springs, with the center node undergoing Brownian random walk due to temperature effects.

## Features

- **1D Mass-Spring Chain**: Linear array of masses connected by springs
- **2D Square Lattice**: Square grid configuration with 4-neighbor connectivity and **periodic boundary conditions**
- **2D Triangular Lattice**: Triangular grid configuration with 6-neighbor connectivity and **periodic boundary conditions**
- **Periodic Boundary Conditions**: 2D lattices wrap around at edges, eliminating boundary effects and preserving translational symmetry
- **Equilibrium Spring Forces**: Springs use Hooke's law with rest length for stable equilibrium positions
- **Brownian Motion**: Center node undergoes temperature-driven random walk
- **Entropy-Driven Migration Tracking**: Measures how non-central masses migrate toward/away from the entropy source
- **Radial Displacement Analysis**: Tracks migration toward center oscillator over time
- **Neighbor Distance Tracking**: Monitors spacing between center and neighboring masses
- **Displacement Tracking**: Tracks and reports displacements of all non-central masses
- **Interactive Configuration**: Configure simulation parameters interactively with sensible defaults
- **Progress Bar**: Real-time progress tracking during simulation execution
- **Enhanced Visualization**: 
  - Shows both initial and final positions
  - Displays displacement vectors as arrows attached to each point
  - Visualizes displacement magnitudes with color-coded heatmaps
  - Migration analysis plots showing attraction/repulsion over time
  - **3D displacement visualization** for 2D networks showing displacement toward center as height
  - **Interactive 3D surface plots** with mouse-controlled rotation and zoom
  - **Live simulation animation** with real-time updates

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CaveroBen/Gravity-Simulator.git
cd Gravity-Simulator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Interactive 3D Visualization Demo (New!)

Experience interactive 3D surface plots and live simulations:

```bash
python demo_interactive.py
```

This will offer two demos:
1. **Interactive 3D Surface Plot**: Rotate and zoom the 3D visualization with your mouse
   - Click and drag to rotate the view
   - Right-click and drag (or scroll) to zoom
   - Points are connected by a smooth triangulated surface
   - Color-coded surface shows displacement patterns
   
2. **Live Simulation**: Watch the simulation evolve in real-time
   - Updates every 50 simulation steps
   - See particles migrate toward the entropy source live
   - Surface plot updates dynamically during simulation

### Multi-Simulation Averaging Demo (New!)

Run the averaging demonstration to see how running multiple simulations and averaging the results provides robust statistical insights:

```bash
python demo_averaging.py
```

This will:
1. Run 10 simulations each for 1D chain, 2D square lattice, and 2D triangular lattice
2. Collect final positions from each simulation run
3. Calculate average positions and displacement vectors across all runs
4. Generate visualizations showing average positions with displacement arrows
5. Provide statistical summaries of averaged results

This feature is useful for:
- Reducing noise from stochastic Brownian motion
- Getting more reliable estimates of system behavior
- Understanding typical displacement patterns
- Statistical analysis of simulation outcomes

### Entropy-Driven Migration Demo (Recommended)

Run the migration demonstration to see how coupled oscillators respond to a center node undergoing Brownian motion:

```bash
python demo_migration.py
```

This will:
1. Run two simulations: baseline (no Brownian) vs. noisy (with Brownian motion)
2. Compare radial displacement (migration toward center)
3. Analyze neighbor distances over time
4. Generate comprehensive comparison visualizations
5. Demonstrate entropy-driven migration in action

### Interactive Mode

Run the example script in interactive mode:

```bash
python example.py
```

You'll be prompted to configure:
- Network type (1d/square/triangular)
- Array size
- Simulation time steps
- Thermal excitation (temperature)
- Spring constant

Simply press Enter to accept the default value for each parameter.

### Batch Mode

To run all three example configurations automatically:

```bash
echo "batch" | python example.py
```

This will:
- Simulate a 1D chain, 2D square lattice, and 2D triangular lattice
- Show progress bars for each simulation
- Generate visualization PNG files with displacement vectors

## Usage

### Basic Example

```python
from quantum_gravity_simulator import Network1D, visualize_network, print_results

# Create a 1D network with 11 masses
network = Network1D(
    n_masses=11,
    mass=1.0,
    spring_constant=10.0,
    damping=0.5,
    dt=0.01,
    temperature=2.0
)

# Run simulation with progress bar
network.simulate(steps=1000, show_progress=True)

# Visualize with displacement vectors
visualize_network(network, "1D Mass-Spring Chain")
```

### 2D Square Lattice

```python
from quantum_gravity_simulator import Network2DSquare

# Create a 5x5 square lattice
network = Network2DSquare(
    size=5,
    mass=1.0,
    spring_constant=8.0,
    damping=0.3,
    dt=0.01,
    temperature=1.5
)

network.simulate(steps=1500, show_progress=True)
```

### 2D Triangular Lattice

```python
from quantum_gravity_simulator import Network2DTriangular

# Create a 5x5 triangular lattice
network = Network2DTriangular(
    size=5,
    mass=1.0,
    spring_constant=8.0,
    damping=0.3,
    dt=0.01,
    temperature=1.5
)

network.simulate(steps=1500, show_progress=True)
```

### 3D Displacement Visualization

For 2D networks, visualize displacement toward the center in 3D:

```python
from quantum_gravity_simulator import Network2DSquare, visualize_network_3d
import matplotlib.pyplot as plt

# Create and simulate a 2D network
network = Network2DSquare(
    size=5,
    mass=1.0,
    spring_constant=8.0,
    damping=0.3,
    dt=0.01,
    temperature=1.5
)

network.simulate(steps=1500, show_progress=True)

# Generate 3D visualization (scatter plot)
fig = visualize_network_3d(network, "3D Displacement View")
plt.savefig('3d_displacement.png', dpi=150, bbox_inches='tight')
plt.show()
```

The 3D visualization displays:
- X and Y coordinates of particle positions
- Z coordinate (height) representing displacement magnitude toward center
- Color-coded points showing displacement intensity
- Red star marking the center node with Brownian noise

### Interactive 3D Surface Visualization

For a more immersive view with points connected by a surface:

```python
from quantum_gravity_simulator import Network2DSquare, visualize_network_3d_surface
import matplotlib.pyplot as plt

# Create and simulate a 2D network
network = Network2DSquare(
    size=5,
    mass=1.0,
    spring_constant=8.0,
    damping=0.3,
    dt=0.01,
    temperature=1.5
)

network.simulate(steps=1500, show_progress=True)

# Generate interactive 3D surface visualization
fig = visualize_network_3d_surface(network, "3D Surface View", interactive=True)
plt.savefig('3d_surface.png', dpi=150, bbox_inches='tight')
plt.show()  # Interactive: click and drag to rotate, scroll to zoom
```

The 3D surface visualization displays:
- X and Y coordinates of particle positions
- Z coordinate (height) representing displacement magnitude toward center
- **Triangulated surface mesh** connecting all particles
- Color-coded surface showing displacement patterns
- Particles overlaid on the surface
- Red star marking the center node with Brownian noise
- **Interactive controls**: Rotate with mouse, zoom with scroll wheel

### Live Simulation Animation

Watch your simulation evolve in real-time with periodic updates:

```python
from quantum_gravity_simulator import Network2DTriangular, animate_simulation_live

# Define network parameters
network_params = {
    'size': 5,
    'mass': 1.0,
    'spring_constant': 8.0,
    'damping': 0.3,
    'dt': 0.01,
    'temperature': 1.5
}

# Run live simulation with updates every 50 steps
network = animate_simulation_live(
    network_class=Network2DTriangular,
    network_params=network_params,
    simulation_steps=1000,
    update_interval=50,  # Update plot every 50 steps
    title="Live Simulation"
)
```

The live simulation:
- Updates the 3D surface plot in real-time
- Shows particle migration as it happens
- Displays current step number in the title
- Allows you to watch entropy-driven dynamics unfold

### Multi-Simulation Averaging

Run multiple simulations and average the results for more robust statistical insights:

```python
from quantum_gravity_simulator import (
    Network2DSquare,
    run_multiple_simulations,
    visualize_averaged_results
)

# Define network parameters
network_params = {
    'size': 5,
    'mass': 1.0,
    'spring_constant': 8.0,
    'damping': 0.3,
    'dt': 0.01,
    'temperature': 1.5
}

# Run 10 simulations
results = run_multiple_simulations(
    network_class=Network2DSquare,
    network_params=network_params,
    simulation_steps=1500,
    n_simulations=10,
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
```

## Visualization Features

The simulator generates comprehensive visualizations showing:

1. **Initial Positions**: Light blue circles showing where masses started
2. **Final Positions**: Blue circles showing where masses ended up
3. **Relative Motion Vectors**: Green arrows showing **relative motion** (motion relative to the grid/center of mass)
   - Vectors are computed by subtracting the mean displacement from each element's displacement
   - This shows the oscillation pattern around the center rather than bulk motion
   - Vectors are **amplified 5x** and drawn with **thicker arrows** for better visibility
4. **Center Node**: Red star marking the thermally-excited center node
5. **Network Connections**: Spring connections shown in both initial (faint gray) and final (blue) configurations
6. **Relative Displacement Heatmap**: Color-coded visualization of relative displacement magnitudes
7. **Migration Analysis Plots**: Time-series showing radial displacement and neighbor distances over time
8. **Averaged Results**: Visualizations of averaged positions and displacements across multiple simulation runs
9. **3D Displacement Visualization**: For 2D networks, a 3D plot where:
   - X and Y axes show spatial positions of particles
   - Z axis (height) represents the magnitude of displacement toward the center node
   - Color-coded points indicate displacement magnitude
   - Center node (with Brownian noise) is highlighted in red
   - Provides intuitive visual understanding of entropy-driven migration patterns
10. **Interactive 3D Surface Visualization** (new): Enhanced 3D visualization with:
   - Triangulated surface mesh connecting all particles
   - Smooth, color-coded surface showing displacement patterns
   - **Interactive controls**: Click and drag to rotate, scroll to zoom
   - Real-time repositioning of the viewing angle
   - Better depth perception with connected surface topology
11. **Live Simulation Animation** (new): Watch simulations evolve in real-time:
   - 3D surface plot updates periodically during simulation
   - Shows particle migration as it happens
   - Configurable update interval for performance tuning
   - Provides dynamic view of entropy-driven dynamics

## Physics Model

The simulator implements a classical mass-spring system with the following physics:

### Spring Forces (with Equilibrium)
Each connection between masses i and j exerts a Hookean restoring force:
```
F = -k * (|r_j - r_i| - r0)
```
where:
- k is the spring constant
- r is the position vector
- r0 is the equilibrium rest length (default: 1.0)

This ensures springs have a stable equilibrium position and don't collapse indefinitely.

### Periodic Boundary Conditions (2D Lattices)
For 2D square and triangular lattices, periodic boundary conditions are applied:
- Nodes at the edge of the grid connect to nodes on the opposite edge
- Distance calculations use the minimum image convention, wrapping around the grid
- For a displacement vector dx between nodes:
  ```
  dx_wrapped = dx - L * round(dx / L)
  ```
  where L is the grid size in each dimension
- This eliminates edge effects and preserves translational symmetry, similar to the mechanism used in `OscillatorArray_Vector.py`

### Damping
Damping forces oppose motion:
```
F_damping = -γ * v
```
where γ is the damping coefficient and v is velocity.

### Brownian Motion and Entropy-Driven Migration
The center node experiences random thermal forces (Brownian motion):
```
δv ~ √(T) * dt * N(0,1)
```
where T is temperature and N(0,1) is standard normal distribution.

**Key behavior:** The center node's random walk creates entropy, and the system naturally migrates towards this entropy source. This is the mechanism that maximizes entropy in a constrained system. Unlike systems with artificial center-of-mass constraints, this simulator allows the array to migrate towards the random-walking element, revealing the fundamental entropy-driven dynamics.

### Time Evolution
Positions are updated using a simple Euler integration:
```
v(t+dt) = v(t) + (F/m) * dt
x(t+dt) = x(t) + v(t) * dt
```

The system exhibits **entropy-driven migration**: as the center element undergoes Brownian motion, other masses naturally migrate towards it through spring force interactions. This emergent behavior demonstrates how entropy maximization drives spatial organization in coupled oscillator systems.

### Migration Tracking and Analysis

The simulator now includes comprehensive tools to measure and analyze entropy-driven migration:

#### Radial Displacement
Measures the mean displacement of non-central masses toward/away from the center oscillator:
```
radial_displacement = distance_initial - distance_current
```
- **Positive values**: Masses have moved toward the center (attraction)
- **Negative values**: Masses have moved away from the center (repulsion)
- Tracked over time to show migration dynamics

#### Neighbor Distance Tracking
Monitors the mean distance from the center oscillator to its immediate neighbors:
- Compares actual distances to equilibrium rest length
- Reveals if coupling causes neighbors to cluster or disperse
- Uses periodic boundary conditions for accurate 2D measurements

#### Comparison Framework
The simulator can compare baseline (no Brownian motion) vs. noisy (with Brownian motion) systems:
- Quantifies the effect of entropy source on system behavior
- Statistical analysis of radial displacement distributions
- Visualization of attraction/repulsion patterns over time

See `demo_migration.py` for a complete demonstration of these analysis capabilities.

### Relative Motion Visualization
The visualizations display **relative motion** rather than absolute motion:
- **Absolute displacement**: The raw change in position from initial to final state
- **Relative displacement**: Displacement minus the mean displacement of all masses
  ```
  d_rel[i] = d_abs[i] - mean(d_abs[all])
  ```
- This removes the bulk translation of the system, showing oscillation patterns relative to the grid
- The center node's Brownian motion causes the system to migrate, and relative motion reveals the patterns of how masses respond to the center's fluctuations

## Parameters

### Network Configuration
- **n_masses** (1D): Number of masses in the chain
- **size** (2D): Size of the lattice (creates size × size grid)

### Physics Parameters
- **mass**: Mass of each node (default: 1.0)
- **spring_constant**: Stiffness of spring connections (default: 1.0)
- **rest_length**: Equilibrium length of springs (default: 1.0)
- **damping**: Damping coefficient (default: 0.1)
- **dt**: Time step for integration (default: 0.01)
- **temperature**: Temperature for Brownian motion (default: 1.0)

### Simulation Parameters
- **steps**: Number of simulation time steps to run
- **show_progress**: Whether to display a progress bar (default: True)

## Output

The simulator provides:

1. **Initial Positions**: Starting coordinates of all masses
2. **Final Positions**: Coordinates of all masses after simulation
3. **Displacements**: Change in position for each mass with displacement vectors
4. **Visualizations**: 
   - For 1D: Position plot with displacement vectors and displacement bar chart
   - For 2D: Network layout showing initial/final positions, displacement vectors, and displacement magnitude heatmap

## Dependencies

- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- tqdm >= 4.62.0 (for progress bars)

## Theory

This simulator is inspired by quantum gravity theories that model spacetime as a discrete network of oscillators. The Brownian motion of the center node represents quantum fluctuations, while the spring connections model the coupling between spacetime elements.

**Entropy-Driven Migration:** When the center element undergoes random walk (Brownian motion), it creates entropy in the system. The key insight is that the mechanism which maximizes entropy in a constrained system (like a mass-spring network) is for the system to migrate towards the entropy source. This simulator demonstrates this fundamental principle by allowing the array to naturally migrate towards the random-walking center element, rather than artificially constraining the center of mass. This emergent behavior reveals how entropy maximization drives spatial organization in coupled systems.

## License

This project is open source and available for educational and research purposes. 
