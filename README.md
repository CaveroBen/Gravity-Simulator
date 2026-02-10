# Quantum Gravity Simulator

A Python package to simulate quantum gravity using mass-spring networks. This simulator models spacetime as arrays (1D or 2D) of masses coupled by springs, with the center node undergoing Brownian random walk due to temperature effects.

## Features

- **1D Mass-Spring Chain**: Linear array of masses connected by springs
- **2D Square Lattice**: Square grid configuration with 4-neighbor connectivity and **periodic boundary conditions**
- **2D Triangular Lattice**: Triangular grid configuration with 6-neighbor connectivity and **periodic boundary conditions**
- **Periodic Boundary Conditions**: 2D lattices wrap around at edges, eliminating boundary effects and preserving translational symmetry
- **Brownian Motion**: Center node undergoes temperature-driven random walk
- **Displacement Tracking**: Tracks and reports displacements of all non-central masses
- **Interactive Configuration**: Configure simulation parameters interactively with sensible defaults
- **Progress Bar**: Real-time progress tracking during simulation execution
- **Enhanced Visualization**: 
  - Shows both initial and final positions
  - Displays displacement vectors as arrows attached to each point
  - Visualizes displacement magnitudes with color-coded heatmaps

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

### Interactive Mode (Default)

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

## Visualization Features

The simulator generates comprehensive visualizations showing:

1. **Initial Positions**: Light blue circles showing where masses started
2. **Final Positions**: Blue circles showing where masses ended up
3. **Displacement Vectors**: Green arrows from initial to final positions
4. **Center Node**: Red star marking the thermally-excited center node
5. **Network Connections**: Spring connections shown in both initial (faint gray) and final (blue) configurations
6. **Displacement Heatmap**: Color-coded visualization of displacement magnitudes

## Physics Model

The simulator implements a classical mass-spring system with the following physics:

### Spring Forces
Each connection between masses i and j exerts a force:
```
F = k * |r_j - r_i|
```
where k is the spring constant and r is the position vector.

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

### Brownian Motion
The center node experiences random thermal forces:
```
δv ~ √(T) * dt * N(0,1)
```
where T is temperature and N(0,1) is standard normal distribution.

### Time Evolution
Positions are updated using a simple Euler integration:
```
v(t+dt) = v(t) + (F/m) * dt
x(t+dt) = x(t) + v(t) * dt
```

## Parameters

### Network Configuration
- **n_masses** (1D): Number of masses in the chain
- **size** (2D): Size of the lattice (creates size × size grid)

### Physics Parameters
- **mass**: Mass of each node (default: 1.0)
- **spring_constant**: Stiffness of spring connections (default: 1.0)
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

This simulator is inspired by quantum gravity theories that model spacetime as a discrete network of oscillators. The Brownian motion of the center node represents quantum fluctuations, while the spring connections model the coupling between spacetime elements. By tracking how these fluctuations propagate through the network, we can study emergent properties of the system.

## License

This project is open source and available for educational and research purposes. 
