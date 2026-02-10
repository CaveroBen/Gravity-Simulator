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
        test_parameter_configuration
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
