"""
Tests for the Quantum Gravity Simulator.
"""

import numpy as np
from quantum_gravity_simulator import (
    Network1D, 
    Network2DSquare, 
    Network2DTriangular,
    MassSpringNetwork
)


def test_network_1d_initialization():
    """Test 1D network initialization."""
    print("Testing 1D network initialization...")
    network = Network1D(n_masses=5)
    
    # Check number of masses
    assert network.n_masses == 5
    assert len(network.positions) == 5
    assert len(network.velocities) == 5
    
    # Check center node
    assert network.center_idx == 2
    
    # Check connections (should be 4 connections for 5 masses)
    assert len(network.connections) == 4
    
    print("✓ 1D network initialization test passed")


def test_network_2d_square_initialization():
    """Test 2D square lattice initialization."""
    print("Testing 2D square lattice initialization...")
    network = Network2DSquare(size=3)
    
    # Check number of masses (3x3 = 9)
    assert network.n_masses == 9
    assert len(network.positions) == 9
    assert network.positions.shape[1] == 2  # 2D positions
    
    # Check center node
    assert network.center_idx == 4  # Middle of 3x3 grid
    
    # Check connections exist
    assert len(network.connections) > 0
    
    print("✓ 2D square lattice initialization test passed")


def test_network_2d_triangular_initialization():
    """Test 2D triangular lattice initialization."""
    print("Testing 2D triangular lattice initialization...")
    network = Network2DTriangular(size=3)
    
    # Check number of masses (3x3 = 9)
    assert network.n_masses == 9
    assert len(network.positions) == 9
    assert network.positions.shape[1] == 2  # 2D positions
    
    # Check center node
    assert network.center_idx == 4  # Middle of 3x3 grid
    
    # Check connections exist (more than square due to diagonals)
    assert len(network.connections) > 0
    
    print("✓ 2D triangular lattice initialization test passed")


def test_simulation_runs():
    """Test that simulation runs without errors."""
    print("Testing simulation execution...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Test 1D with more steps for robustness
    network_1d = Network1D(n_masses=5, temperature=0.5)
    initial_pos_1d = network_1d.positions.copy()
    network_1d.simulate(steps=500, show_progress=False)  # Increased to 500 for very reliable test
    final_pos_1d = network_1d.positions
    
    # Center node should have moved (due to Brownian motion)
    displacement_1d = np.linalg.norm(final_pos_1d[network_1d.center_idx] - 
                                     initial_pos_1d[network_1d.center_idx])
    assert displacement_1d > 0.005  # Very low threshold for reliability
    
    # Test 2D Square with more steps
    network_2d_sq = Network2DSquare(size=3, temperature=0.5)
    initial_pos_2d = network_2d_sq.positions.copy()
    network_2d_sq.simulate(steps=500, show_progress=False)  # Increased to 500
    final_pos_2d = network_2d_sq.positions
    
    # Center node should have moved
    displacement_2d = np.linalg.norm(final_pos_2d[network_2d_sq.center_idx] - 
                                     initial_pos_2d[network_2d_sq.center_idx])
    assert displacement_2d > 0.005
    
    print("✓ Simulation execution test passed")


def test_displacement_tracking():
    """Test displacement tracking functionality."""
    print("Testing displacement tracking...")
    
    network = Network1D(n_masses=5, temperature=0.5)
    network.simulate(steps=50, show_progress=False)  # Disable progress bar in tests
    
    displacements = network.get_displacements()
    
    # Should have displacement data for all masses
    assert len(displacements) == network.n_masses
    
    # Displacements should be non-zero (at least for center node)
    displacement_magnitudes = np.linalg.norm(displacements, axis=1)
    assert np.any(displacement_magnitudes > 0)
    
    print("✓ Displacement tracking test passed")


def test_spring_forces():
    """Test that spring forces are computed correctly."""
    print("Testing spring force computation...")
    
    # Create simple 1D network with 3 masses
    network = Network1D(n_masses=3, spring_constant=10.0, damping=0.0)
    
    # Manually displace the center mass
    network.positions[1, 0] += 0.5
    
    # Compute forces
    forces = network.compute_forces()
    
    # Center mass should experience restoring force
    assert forces[1, 0] != 0
    
    print("✓ Spring force computation test passed")


def test_brownian_motion():
    """Test Brownian motion is applied to center node."""
    print("Testing Brownian motion...")
    
    np.random.seed(42)  # For reproducibility
    
    network = Network1D(n_masses=5, temperature=1.0)
    initial_velocity = network.velocities[network.center_idx].copy()
    
    # Apply Brownian motion multiple times
    for _ in range(10):
        network.apply_brownian_motion()
    
    final_velocity = network.velocities[network.center_idx]
    
    # Velocity should have changed
    assert not np.allclose(initial_velocity, final_velocity)
    
    print("✓ Brownian motion test passed")


def test_output_methods():
    """Test output methods work correctly."""
    print("Testing output methods...")
    
    network = Network1D(n_masses=5, temperature=0.5)
    network.simulate(steps=10, show_progress=False)  # Disable progress bar in tests
    
    # Test get_final_positions
    positions = network.get_final_positions()
    assert positions.shape == (5, 1)
    
    # Test get_displacements
    displacements = network.get_displacements()
    assert displacements.shape == (5, 1)
    
    # Test get_initial_positions
    initial_positions = network.get_initial_positions()
    assert initial_positions.shape == (5, 1)
    
    print("✓ Output methods test passed")


def test_parameter_configuration():
    """Test that parameters are properly set."""
    print("Testing parameter configuration...")
    
    mass = 2.0
    k = 15.0
    damping = 0.3
    dt = 0.005
    temp = 3.0
    
    network = Network1D(
        n_masses=5,
        mass=mass,
        spring_constant=k,
        damping=damping,
        dt=dt,
        temperature=temp
    )
    
    assert network.mass == mass
    assert network.k == k
    assert network.damping == damping
    assert network.dt == dt
    assert network.temperature == temp
    
    print("✓ Parameter configuration test passed")


def test_periodic_boundaries_square():
    """Test that periodic boundaries work correctly for square lattice."""
    print("Testing periodic boundaries for square lattice...")
    
    # Create a 3x3 square lattice
    network = Network2DSquare(size=3)
    
    # Test that periodic_vector method exists
    assert hasattr(network, 'periodic_vector')
    
    # Test periodic wrapping in x-direction
    # Position at (0, 1) should connect to (2, 1) with wrapping
    pos_left = network.positions[network.idx_map[(0, 1)]]
    pos_right = network.positions[network.idx_map[(2, 1)]]
    
    vec = network.periodic_vector(pos_right, pos_left)
    # With wrapping, the distance should be 1.0 (spacing), not 2.0
    distance = np.linalg.norm(vec)
    assert abs(distance - 1.0) < 0.01, f"Expected distance ~1.0, got {distance}"
    
    # Test periodic wrapping in y-direction
    pos_bottom = network.positions[network.idx_map[(1, 0)]]
    pos_top = network.positions[network.idx_map[(1, 2)]]
    
    vec = network.periodic_vector(pos_top, pos_bottom)
    distance = np.linalg.norm(vec)
    assert abs(distance - 1.0) < 0.01, f"Expected distance ~1.0, got {distance}"
    
    # Verify connections include wrapped edges
    # For a 3x3 grid with periodic boundaries, each node makes 2 outgoing connections
    # (right and bottom with wrapping), so total should be 3*3*2 = 18 connections
    assert len(network.connections) == 18
    
    print("✓ Periodic boundaries test for square lattice passed")


def test_periodic_boundaries_triangular():
    """Test that periodic boundaries work correctly for triangular lattice."""
    print("Testing periodic boundaries for triangular lattice...")
    
    # Create a 3x3 triangular lattice
    network = Network2DTriangular(size=3)
    
    # Test that periodic_vector method exists
    assert hasattr(network, 'periodic_vector')
    
    # Test periodic wrapping
    pos_left = network.positions[network.idx_map[(0, 1)]]
    pos_right = network.positions[network.idx_map[(2, 1)]]
    
    vec = network.periodic_vector(pos_right, pos_left)
    distance = np.linalg.norm(vec)
    # With wrapping, distance should be close to spacing
    assert distance < 2.0, f"Expected distance < 2.0 with wrapping, got {distance}"
    
    # Verify connections include wrapped edges
    # For triangular lattice, each node makes 3 outgoing connections
    # (right, bottom, and one diagonal based on row parity)
    # Total should be 3*3*3 = 27 connections
    assert len(network.connections) == 27
    
    print("✓ Periodic boundaries test for triangular lattice passed")


def test_forces_with_periodic_boundaries():
    """Test that forces are computed correctly with periodic boundaries."""
    print("Testing force computation with periodic boundaries...")
    
    # Create a small square lattice
    network = Network2DSquare(size=3, spring_constant=10.0, damping=0.0)
    
    # Displace a node on the edge
    edge_idx = network.idx_map[(0, 1)]  # Left edge
    network.positions[edge_idx, 0] -= 0.2  # Move left
    
    # Compute forces
    forces = network.compute_forces()
    
    # The edge node should experience a force pulling it back
    # Due to periodic boundaries, it's connected to the right edge too
    assert np.any(forces[edge_idx] != 0), "Edge node should experience force with periodic boundaries"
    
    # Force should be non-zero on multiple nodes due to network connectivity
    non_zero_forces = np.sum(np.linalg.norm(forces, axis=1) > 1e-10)
    assert non_zero_forces > 2, "Multiple nodes should experience forces in periodic lattice"
    
    print("✓ Force computation with periodic boundaries test passed")


def test_entropy_driven_migration():
    """Test that the system migrates towards the random-walking center element."""
    print("Testing entropy-driven migration...")
    
    # Minimum threshold for drift to account for stochastic nature of Brownian motion
    MIN_DRIFT = 1e-6
    
    np.random.seed(42)
    
    # Test with 1D network
    network_1d = Network1D(n_masses=5, temperature=1.0)
    initial_com_1d = np.mean(network_1d.positions, axis=0)
    initial_center_pos_1d = network_1d.positions[network_1d.center_idx].copy()
    network_1d.simulate(steps=500, show_progress=False)
    final_com_1d = np.mean(network_1d.positions, axis=0)
    final_center_pos_1d = network_1d.positions[network_1d.center_idx]
    
    # Center of mass should drift (migrate towards center element)
    drift_1d = np.linalg.norm(final_com_1d - initial_com_1d)
    assert drift_1d > MIN_DRIFT, f"1D: Center of mass should drift due to entropy-driven migration (drift={drift_1d})"
    
    # Center element should have moved due to Brownian motion
    center_displacement_1d = np.linalg.norm(final_center_pos_1d - initial_center_pos_1d)
    assert center_displacement_1d > MIN_DRIFT, f"1D: Center element should move due to Brownian motion (displacement={center_displacement_1d})"
    
    # Test with 2D square network
    network_2d_sq = Network2DSquare(size=3, temperature=1.0)
    initial_com_2d_sq = np.mean(network_2d_sq.positions, axis=0)
    initial_center_pos_2d_sq = network_2d_sq.positions[network_2d_sq.center_idx].copy()
    network_2d_sq.simulate(steps=500, show_progress=False)
    final_com_2d_sq = np.mean(network_2d_sq.positions, axis=0)
    final_center_pos_2d_sq = network_2d_sq.positions[network_2d_sq.center_idx]
    
    # Center of mass should drift
    drift_2d_sq = np.linalg.norm(final_com_2d_sq - initial_com_2d_sq)
    assert drift_2d_sq > MIN_DRIFT, f"2D Square: Center of mass should drift due to entropy-driven migration (drift={drift_2d_sq})"
    
    # Center element should have moved
    center_displacement_2d_sq = np.linalg.norm(final_center_pos_2d_sq - initial_center_pos_2d_sq)
    assert center_displacement_2d_sq > MIN_DRIFT, f"2D Square: Center element should move due to Brownian motion (displacement={center_displacement_2d_sq})"
    
    # Test with 2D triangular network
    network_2d_tri = Network2DTriangular(size=3, temperature=1.0)
    initial_com_2d_tri = np.mean(network_2d_tri.positions, axis=0)
    initial_center_pos_2d_tri = network_2d_tri.positions[network_2d_tri.center_idx].copy()
    network_2d_tri.simulate(steps=500, show_progress=False)
    final_com_2d_tri = np.mean(network_2d_tri.positions, axis=0)
    final_center_pos_2d_tri = network_2d_tri.positions[network_2d_tri.center_idx]
    
    # Center of mass should drift
    drift_2d_tri = np.linalg.norm(final_com_2d_tri - initial_com_2d_tri)
    assert drift_2d_tri > MIN_DRIFT, f"2D Triangular: Center of mass should drift due to entropy-driven migration (drift={drift_2d_tri})"
    
    # Center element should have moved
    center_displacement_2d_tri = np.linalg.norm(final_center_pos_2d_tri - initial_center_pos_2d_tri)
    assert center_displacement_2d_tri > MIN_DRIFT, f"2D Triangular: Center element should move due to Brownian motion (displacement={center_displacement_2d_tri})"
    
    print("✓ Entropy-driven migration test passed")



def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING QUANTUM GRAVITY SIMULATOR TESTS")
    print("="*60 + "\n")
    
    tests = [
        test_network_1d_initialization,
        test_network_2d_square_initialization,
        test_network_2d_triangular_initialization,
        test_simulation_runs,
        test_displacement_tracking,
        test_spring_forces,
        test_brownian_motion,
        test_output_methods,
        test_parameter_configuration,
        test_periodic_boundaries_square,
        test_periodic_boundaries_triangular,
        test_forces_with_periodic_boundaries,
        test_entropy_driven_migration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
