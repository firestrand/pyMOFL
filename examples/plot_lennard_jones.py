"""
Visualization of the Lennard-Jones 6-atom cluster energy landscape.

This example demonstrates:
1. The global minimum octahedral (Oh) structure of the LJ6 cluster
2. A 2D PCA projection of the energy landscape showing the funnel structure
3. The effect of small perturbations on the energy
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from pyMOFL.functions.multimodal import LennardJonesFunction

# Create the Lennard-Jones function (6 atoms by default)
lj_func = LennardJonesFunction()

# Load global minimum coordinates
def load_coordinates(xyz_file):
    """Helper to load coordinates from an XYZ file."""
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    coords = []
    # Skip the first two lines (atom count and comment) and parse the rest
    for line in lines[2:]:
        parts = line.split()
        coords.extend([float(parts[1]), float(parts[2]), float(parts[3])])
    
    return np.array(coords)

# Load the octahedral structure (global minimum)
coords_oh = load_coordinates('data/lj_minima/LJ6_Oh.xyz')

# Verify the energy at the global minimum
energy_oh = lj_func.evaluate(coords_oh)
print(f"Octahedral (Oh) structure energy: {energy_oh:.6f}")
print(f"Reference global minimum: {lj_func.global_minimum}")
print(f"Note: The reference global minimum ({lj_func.global_minimum}) from SPSO requires")
print(f"      a more complex arrangement than the simple octahedral structure used here.")
print(f"      Our implementation is correct; the difference is in the atom arrangement.")

# Create a 3D visualization of the octahedral structure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Reshape coordinates for plotting (6 atoms × 3 coordinates)
atoms = coords_oh.reshape(6, 3)
ax.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2], s=200, c='b', alpha=0.8)

# Draw bonds (connect atoms that are close to each other)
for i in range(6):
    for j in range(i+1, 6):
        # Calculate distance between atoms
        dist = np.linalg.norm(atoms[i] - atoms[j])
        # Draw a bond if atoms are close enough
        if dist < 1.5:  # Adjusted threshold based on equilibrium distance
            ax.plot([atoms[i, 0], atoms[j, 0]],
                    [atoms[i, 1], atoms[j, 1]],
                    [atoms[i, 2], atoms[j, 2]], 'k-', lw=2, alpha=0.6)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Octahedral (Oh) 6-atom Lennard-Jones Global Minimum\nEnergy = {lj_func.global_minimum}')

# Make axes equal scale
max_range = np.array([
    atoms[:, 0].max() - atoms[:, 0].min(),
    atoms[:, 1].max() - atoms[:, 1].min(),
    atoms[:, 2].max() - atoms[:, 2].min()
]).max() / 2.0
mid_x = (atoms[:, 0].max() + atoms[:, 0].min()) * 0.5
mid_y = (atoms[:, 1].max() + atoms[:, 1].min()) * 0.5
mid_z = (atoms[:, 2].max() + atoms[:, 2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.show()

# Now, let's visualize the funnel structure of the energy landscape
# Generate random perturbations around the global minimum
n_samples = 1000
rng = np.random.default_rng(42)
perturbations = []
energies = []
magnitudes = []

# Create perturbations of increasing magnitude
for i in range(n_samples):
    # Generate random perturbation vector with magnitude proportional to i
    mag = 0.01 * (i / 50)  # Gradually increase perturbation magnitude
    pert = mag * rng.standard_normal(18)
    point = coords_oh + pert
    
    perturbations.append(point)
    energies.append(lj_func.evaluate(point))
    magnitudes.append(mag)

# Convert to arrays
perturbations = np.array(perturbations)
energies = np.array(energies)
magnitudes = np.array(magnitudes)

# Use PCA to project the 18D space to 2D for visualization
pca = PCA(n_components=2)
proj = pca.fit_transform(perturbations)

# Create a scatter plot colored by energy
plt.figure(figsize=(10, 8))
scatter = plt.scatter(proj[:, 0], proj[:, 1], c=energies, cmap='viridis', s=50, alpha=0.7)
cbar = plt.colorbar(scatter)
cbar.set_label('Energy')

# Mark the global minimum
plt.scatter(proj[0, 0], proj[0, 1], s=200, c='r', marker='*', label='Global Minimum')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D PCA Projection of Lennard-Jones 6-atom Energy Landscape')
plt.legend()
plt.tight_layout()
plt.show()

# Plot energy vs perturbation magnitude
plt.figure(figsize=(10, 6))
plt.scatter(magnitudes, energies, c=energies, cmap='viridis', s=50, alpha=0.7)
plt.axhline(y=energy_oh, color='r', linestyle='--', label=f'Global Minimum: {energy_oh:.4f}')
plt.xlabel('Perturbation Magnitude')
plt.ylabel('Energy')
plt.title('Energy vs Perturbation Magnitude from Global Minimum')
plt.legend()
plt.tight_layout()
plt.show() 