"""
Test script for interactive 3D visualization features.

This script tests:
1. visualize_network_3d_surface function with surface plots
2. animate_simulation_live function (in non-interactive mode)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from quantum_gravity_simulator import (
    Network2DSquare,
    Network2DTriangular,
    visualize_network_3d_surface,
    animate_simulation_live
)


def test_surface_visualization():
    """Test the surface visualization function."""
    print("\n" + "="*70)
    print("TEST 1: Surface Visualization")
    print("="*70)
    
    np.random.seed(42)
    
    # Test with square lattice
    print("\nTesting with 2D square lattice...")
    network = Network2DSquare(
        size=4,
        mass=1.0,
        spring_constant=8.0,
        damping=0.3,
        dt=0.01,
        temperature=1.5
    )
    
    network.simulate(steps=500, show_progress=True)
    
    fig = visualize_network_3d_surface(
        network,
        title="Test Square Lattice - 3D Surface",
        interactive=False
    )
    
    if fig is not None:
        plt.savefig('test_square_surface.png', dpi=150, bbox_inches='tight')
        print("✓ Square lattice surface visualization successful")
        print("  Saved to: test_square_surface.png")
        plt.close()
    else:
        print("✗ Square lattice surface visualization failed")
        return False
    
    # Test with triangular lattice
    print("\nTesting with 2D triangular lattice...")
    network = Network2DTriangular(
        size=4,
        mass=1.0,
        spring_constant=8.0,
        damping=0.3,
        dt=0.01,
        temperature=1.5
    )
    
    network.simulate(steps=500, show_progress=True)
    
    fig = visualize_network_3d_surface(
        network,
        title="Test Triangular Lattice - 3D Surface",
        interactive=False
    )
    
    if fig is not None:
        plt.savefig('test_triangular_surface.png', dpi=150, bbox_inches='tight')
        print("✓ Triangular lattice surface visualization successful")
        print("  Saved to: test_triangular_surface.png")
        plt.close()
    else:
        print("✗ Triangular lattice surface visualization failed")
        return False
    
    print("\n" + "="*70)
    print("✓ All surface visualization tests passed!")
    print("="*70)
    return True


def test_live_simulation():
    """Test the live simulation function (in non-interactive mode)."""
    print("\n" + "="*70)
    print("TEST 2: Live Simulation (Fast Mode)")
    print("="*70)
    
    np.random.seed(123)
    
    # Use smaller network and fewer steps for faster testing
    network_params = {
        'size': 3,
        'mass': 1.0,
        'spring_constant': 8.0,
        'damping': 0.3,
        'dt': 0.01,
        'temperature': 1.5
    }
    
    print("\nRunning live simulation with 2D square lattice...")
    print("(Using non-interactive mode for testing)")
    
    try:
        network = animate_simulation_live(
            network_class=Network2DSquare,
            network_params=network_params,
            simulation_steps=200,  # Fewer steps for testing
            update_interval=50,
            title="Test Live Simulation"
        )
        
        # Verify the network was created and simulated
        if network is not None and hasattr(network, 'positions'):
            print("✓ Live simulation completed successfully")
            
            # Generate final visualization
            fig = visualize_network_3d_surface(
                network,
                title="Test Live Simulation - Final State",
                interactive=False
            )
            
            if fig is not None:
                plt.savefig('test_live_simulation_final.png', dpi=150, bbox_inches='tight')
                print("  Saved final state to: test_live_simulation_final.png")
                plt.close()
            
            print("\n" + "="*70)
            print("✓ Live simulation test passed!")
            print("="*70)
            return True
        else:
            print("✗ Live simulation returned invalid network")
            return False
            
    except Exception as e:
        print(f"✗ Live simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("*" * 70)
    print("INTERACTIVE FEATURES TEST SUITE")
    print("*" * 70)
    
    results = []
    
    # Run tests
    results.append(("Surface Visualization", test_surface_visualization()))
    results.append(("Live Simulation", test_live_simulation()))
    
    # Summary
    print("\n")
    print("*" * 70)
    print("TEST SUMMARY")
    print("*" * 70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70 + "\n")
        return 0
    else:
        print("\n" + "="*70)
        print("✗ SOME TESTS FAILED")
        print("="*70 + "\n")
        return 1


if __name__ == "__main__":
    exit(main())
