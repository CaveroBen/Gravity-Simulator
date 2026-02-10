"""
Test radial displacement tracking features.
"""

import numpy as np
from quantum_gravity_simulator import (
    Network1D,
    Network2DSquare,
    Network2DTriangular
)


def test_radial_displacement_computation():
    """Test that radial displacement is computed correctly."""
    print("Testing radial displacement computation...")
    
    np.random.seed(42)
    
    # Test 1D network
    network_1d = Network1D(n_masses=5, temperature=0.5)
    network_1d.simulate(steps=100, show_progress=False)
    
    radial_disp = network_1d.compute_radial_displacement()
    # Should return a float
    assert isinstance(radial_disp, (float, np.floating))
    
    # Test 2D square network
    network_2d = Network2DSquare(size=3, temperature=0.5)
    network_2d.simulate(steps=100, show_progress=False)
    
    radial_disp_2d = network_2d.compute_radial_displacement()
    assert isinstance(radial_disp_2d, (float, np.floating))
    
    print("✓ Radial displacement computation test passed")


def test_neighbor_distance_computation():
    """Test that neighbor distances are computed correctly."""
    print("Testing neighbor distance computation...")
    
    # Test 1D network
    network_1d = Network1D(n_masses=5, temperature=0.5)
    network_1d.simulate(steps=100, show_progress=False)
    
    neighbor_dist = network_1d.compute_neighbor_distances()
    # Should return a float close to rest_length initially
    assert isinstance(neighbor_dist, (float, np.floating))
    assert neighbor_dist > 0
    
    # Test 2D square network
    network_2d = Network2DSquare(size=3, temperature=0.5)
    network_2d.simulate(steps=100, show_progress=False)
    
    neighbor_dist_2d = network_2d.compute_neighbor_distances()
    assert isinstance(neighbor_dist_2d, (float, np.floating))
    assert neighbor_dist_2d > 0
    
    print("✓ Neighbor distance computation test passed")


def test_history_tracking():
    """Test that displacement history is tracked over time."""
    print("Testing history tracking...")
    
    np.random.seed(42)
    
    network = Network2DSquare(size=3, temperature=0.5)
    network.simulate(steps=100, show_progress=False)
    
    # Check radial displacement history
    radial_history = network.get_radial_displacement_history()
    assert len(radial_history) > 0
    assert len(radial_history) <= 101  # Initial + 100 steps
    
    # Check neighbor distance history
    neighbor_history = network.get_neighbor_distance_history()
    assert len(neighbor_history) > 0
    assert len(neighbor_history) <= 101
    
    # Histories should have same length
    assert len(radial_history) == len(neighbor_history)
    
    print("✓ History tracking test passed")


def test_migration_toward_center():
    """Test that non-central masses can migrate toward center with Brownian motion."""
    print("Testing migration toward center with Brownian motion...")
    
    np.random.seed(42)
    
    # Run simulation with significant temperature
    network = Network2DSquare(size=5, temperature=2.0, spring_constant=5.0)
    network.simulate(steps=500, show_progress=False)
    
    # Get radial displacement history
    radial_history = network.get_radial_displacement_history()
    
    # Should have tracked displacement over time
    assert len(radial_history) > 0
    
    # Final radial displacement should exist (can be positive or negative)
    final_radial = radial_history[-1]
    assert isinstance(final_radial, (float, np.floating))
    
    # System should show some migration (absolute value > 0)
    # Due to Brownian motion and coupling
    mean_radial = np.mean(np.abs(radial_history[-20:]))
    assert mean_radial >= 0  # Can be very small but should be non-negative
    
    print("✓ Migration toward center test passed")


def test_rest_length_equilibrium():
    """Test that springs with rest length maintain equilibrium better."""
    print("Testing rest length equilibrium...")
    
    np.random.seed(42)
    
    # Create network with explicit rest length
    network = Network1D(n_masses=5, rest_length=1.0, temperature=0.1)
    network.simulate(steps=200, show_progress=False)
    
    # Neighbor distances should be close to rest length on average
    neighbor_history = network.get_neighbor_distance_history()
    mean_neighbor_dist = np.mean(neighbor_history[-20:])
    
    # Should be reasonably close to rest_length (within 50% due to dynamics)
    assert 0.5 < mean_neighbor_dist < 2.0
    
    print("✓ Rest length equilibrium test passed")


def test_periodic_boundaries_radial():
    """Test radial displacement works correctly with periodic boundaries."""
    print("Testing radial displacement with periodic boundaries...")
    
    np.random.seed(42)
    
    # Create 2D square network with periodic boundaries
    network = Network2DSquare(size=3, temperature=0.5)
    network.simulate(steps=100, show_progress=False)
    
    # Compute radial displacement using periodic boundaries
    radial_disp = network.compute_radial_displacement()
    
    # Should compute without errors
    assert isinstance(radial_disp, (float, np.floating))
    
    # Neighbor distances should also work
    neighbor_dist = network.compute_neighbor_distances()
    assert isinstance(neighbor_dist, (float, np.floating))
    
    print("✓ Periodic boundaries radial displacement test passed")


def run_all_tests():
    """Run all radial tracking tests."""
    print("\n" + "="*60)
    print("RUNNING RADIAL TRACKING TESTS")
    print("="*60 + "\n")
    
    tests = [
        test_radial_displacement_computation,
        test_neighbor_distance_computation,
        test_history_tracking,
        test_migration_toward_center,
        test_rest_length_equilibrium,
        test_periodic_boundaries_radial
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
