# Quantum Gravity Simulator

A Python package to simulate quantum gravity using mass-spring networks. This simulator models spacetime as arrays (1D or 2D) of masses coupled by springs, with the center node undergoing Brownian random walk due to temperature effects.

## Features

- **1D Mass-Spring Chain**: Linear array of masses connected by springs
- **2D Square Lattice**: Square grid configuration with 4-neighbor connectivity
- **2D Triangular Lattice**: Triangular grid configuration with 6-neighbor connectivity
- **Brownian Motion**: Center node undergoes temperature-driven random walk
- **Displacement Tracking**: Tracks and reports displacements of all non-central masses
- **Visualization**: Generates plots showing final positions and displacement magnitudes

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

Run the example script to see all three configurations in action:

```bash
python example.py
```

This will:
- Simulate a 1D chain, 2D square lattice, and 2D triangular lattice
- Print detailed results for each simulation
- Generate visualization PNG files

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

# Run simulation
network.simulate(steps=1000)

# Print results
print_results(network)

# Visualize
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

network.simulate(steps=1500)
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

network.simulate(steps=1500)
```

## Physics Model

The simulator implements a classical mass-spring system with the following physics:

### Spring Forces
Each connection between masses i and j exerts a force:
```
F = k * |r_j - r_i|
```
where k is the spring constant and r is the position vector.

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

- **mass**: Mass of each node (default: 1.0)
- **spring_constant**: Stiffness of spring connections (default: 1.0)
- **damping**: Damping coefficient (default: 0.1)
- **dt**: Time step for integration (default: 0.01)
- **temperature**: Temperature for Brownian motion (default: 1.0)

## Output

The simulator provides:

1. **Final Positions**: Coordinates of all masses after simulation
2. **Displacements**: Change in position for each non-central mass
3. **Statistics**: Mean, standard deviation, min, and max displacement magnitudes
4. **Visualizations**: 
   - For 1D: Position plot and displacement bar chart
   - For 2D: Network layout with connections and displacement heatmap

## Dependencies

- NumPy >= 1.20.0
- Matplotlib >= 3.3.0

## Theory

This simulator is inspired by quantum gravity theories that model spacetime as a discrete network of oscillators. The Brownian motion of the center node represents quantum fluctuations, while the spring connections model the coupling between spacetime elements. By tracking how these fluctuations propagate through the network, we can study emergent properties of the system.

## License

This project is open source and available for educational and research purposes. 
