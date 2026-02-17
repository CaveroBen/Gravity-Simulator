# Radial Displacement Projection Method

## Overview

The radial displacement tracking has been updated to use **vector projection** instead of distance change. This provides a more accurate measurement of motion toward or away from the central (thermal) node.

## What Changed

### Previous Implementation (Distance-Based)

The old method calculated radial displacement as:
```python
dist_initial = ||center_initial - node_initial||
dist_current = ||center_current - node_current||
radial_displacement = dist_initial - dist_current
```

**Problem**: This measures the change in straight-line distance, which is affected by both radial AND tangential motion components.

### New Implementation (Projection-Based)

The new method uses vector projection:
```python
displacement = node_current - node_initial
direction_to_center = (center_initial - node_initial) / ||center_initial - node_initial||
radial_displacement = displacement · direction_to_center
```

**Advantage**: This captures only the component of displacement that is directed toward/away from the center, using the **undisplaced center position** as reference.

## Key Benefits

1. **Accurate radial component**: Only measures motion in the direction of the center
2. **Filters tangential motion**: Motion perpendicular to the radial direction contributes zero
3. **Uses undisplaced reference**: Center's initial position is used for direction calculation
4. **Handles mixed motion correctly**: Extracts radial component from complex displacement patterns

## Mathematical Details

For each non-central node `i`:

1. **Displacement vector**: 
   ```
   d_i = position_current[i] - position_initial[i]
   ```

2. **Direction to center** (from initial position):
   ```
   r_i = center_initial - position_initial[i]
   r̂_i = r_i / ||r_i||
   ```

3. **Radial projection**:
   ```
   radial_i = d_i · r̂_i
   ```

4. **Mean radial displacement**:
   ```
   radial_mean = (1/N) Σ radial_i
   ```

Where:
- Positive value = motion toward center (attraction)
- Negative value = motion away from center (repulsion)
- Zero = no net radial motion (or pure tangential motion)

## Comparison Examples

### Example 1: Pure Tangential Motion

A node moves perpendicular to the radial direction:

```
Initial position:     [0, 0]
Center position:      [1, 1]
Radial direction:     [0.707, 0.707] (normalized)
Tangential direction: [-0.707, 0.707] (perpendicular)

Movement: 0.3 units tangentially

Old method (distance): -0.031 (incorrect - shows radial motion)
New method (projection): 0.000 (correct - identifies non-radial motion)
```

### Example 2: Mixed Radial + Tangential Motion

A node moves both toward center and tangentially:

```
Radial component:     +0.2 (toward center)
Tangential component: +0.3 (perpendicular)

Old method (distance): +0.163 (affected by both components)
New method (projection): +0.200 (correctly extracts radial component)
```

### Example 3: Pure Radial Motion

A node moves directly toward or away from center:

```
Movement: -0.15 (away from center)

Old method (distance): -0.150 (correct)
New method (projection): -0.150 (correct - both methods agree)
```

## Visual Demonstration

The visualization shows:

1. **Initial Configuration**: Starting positions with center marked in red
2. **Final Configuration**: Final positions with displacement vectors (gray arrows)
3. **Radial Projection**: Only the radial components shown (green = toward, orange = away)
4. **Distribution**: Histogram of projection values with statistics

![Projection Method Visualization](projection_method_visualization.png)

## Code Location

The implementation is in `quantum_gravity_simulator.py`:

```python
def compute_radial_displacement(self) -> float:
    """
    Compute mean displacement of non-central masses toward/away from the center node.
    
    Uses vector projection of displacement onto the direction towards the initial
    (undisplaced) central node position. This captures the component of motion
    that is specifically directed toward or away from the thermal center.
    ...
    """
```

## Testing

All tests pass with the new implementation:
- ✅ 6/6 radial tracking tests
- ✅ 16/16 main simulator tests
- ✅ Projection correctly identifies tangential vs radial motion
- ✅ Works with periodic boundary conditions

## Physical Interpretation

In the context of the quantum gravity simulator:

- **Thermal center node**: Receives additional energy (higher temperature)
- **Other nodes**: Should be attracted toward the thermal center due to entropy-driven migration
- **Radial projection**: Measures how much each node actually moves toward/away from this thermal source
- **Mean projection**: Indicates net migration behavior of the system

This projection method provides a clearer physical picture of the thermal-driven migration process by filtering out irrelevant tangential oscillations and focusing on the directional component toward the energy source.

## References

The requirement specified:
> "The displacement needs to be recorded in the direction of the central (thermal) node. Take the projection of the displacement in the direction of the central node using the position of the undisplaced central node."

This implementation fulfills that requirement by:
1. Computing displacement vectors for each node
2. Projecting them onto the direction toward the center
3. Using the undisplaced (initial) center position as the reference direction
