"""
Visualization of the Tripod function (SPSO ID 4).

This example creates a 3D surface plot and a 2D contour plot
of the Tripod function to illustrate its discontinuous, 
multimodal nature with ridges along the coordinate axes.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyMOFL.functions.multimodal import TripodFunction

# Create the Tripod function
tripod = TripodFunction()

# Create a grid of points for visualization
x1 = np.linspace(-100, 100, 400)
x2 = np.linspace(-100, 100, 400)
X1, X2 = np.meshgrid(x1, x2)

# Create a batch of points for evaluation
points = np.column_stack((X1.ravel(), X2.ravel()))

# Evaluate function at all points
Z = tripod.evaluate_batch(points)

# Reshape the results for plotting
Z = Z.reshape(X1.shape)

# Create a figure with two subplots
fig = plt.figure(figsize=(16, 8))

# 3D Surface plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
# Clip Z values to make the surface plot more readable
Z_clipped = np.clip(Z, 0, 200)
surf = ax1.plot_surface(X1, X2, Z_clipped, cmap='viridis', alpha=0.8, antialiased=True)
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_zlabel('$f(x_1, x_2)$')
ax1.set_title('Tripod Function - 3D Surface')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

# Zoom in on the region of interest for better visibility
ax1.set_xlim([-60, 60])
ax1.set_ylim([-60, 60])
ax1.set_zlim([0, 100])

# Create a contour plot
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X1, X2, Z, levels=50, cmap='viridis')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_title('Tripod Function - Contour Plot')
fig.colorbar(contour, ax=ax2)

# Mark the global minimum
ax2.plot(0, -50, 'rx', markersize=10, label='Global Minimum (0, -50)')
ax2.legend()

# Zoom in on the region of interest
ax2.set_xlim([-60, 60])
ax2.set_ylim([-60, 60])

# Add grid lines to show the axes
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.suptitle('Tripod Function Visualization', fontsize=16)
plt.subplots_adjust(top=0.9)

plt.show()

# To create a zoomed-in plot around the global minimum
plt.figure(figsize=(8, 6))
# Create a finer grid near the global minimum
x1_zoom = np.linspace(-10, 10, 200)
x2_zoom = np.linspace(-60, -40, 200)
X1_zoom, X2_zoom = np.meshgrid(x1_zoom, x2_zoom)
points_zoom = np.column_stack((X1_zoom.ravel(), X2_zoom.ravel()))
Z_zoom = tripod.evaluate_batch(points_zoom).reshape(X1_zoom.shape)

# Plot the zoomed-in contour
plt.contourf(X1_zoom, X2_zoom, Z_zoom, levels=20, cmap='viridis')
plt.colorbar(label='$f(x_1, x_2)$')
plt.plot(0, -50, 'rx', markersize=10)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Tripod Function - Zoomed at Global Minimum (0, -50)')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.tight_layout()

plt.show() 