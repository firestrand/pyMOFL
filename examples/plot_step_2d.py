"""
Visualization of the Step function (SPSO ID 15).

This example creates a 2D visualization of the Step function by fixing 8 dimensions
at zero and varying 2 dimensions to illustrate the function's discontinuous,
flat plateau landscape.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyMOFL.functions.multimodal import StepFunction

# Create the Step function
step_func = StepFunction(bias=30.0)

# Create a grid of points for 2D visualization
resolution = 200
x1 = np.linspace(-5, 5, resolution)
x2 = np.linspace(-5, 5, resolution)
X1, X2 = np.meshgrid(x1, x2)

# Create an array to store function values
Z = np.zeros_like(X1)

# Evaluate function at each grid point, keeping other dimensions fixed at 0
for i in range(resolution):
    for j in range(resolution):
        # Create a 10D point with only the first two dimensions non-zero
        x = np.zeros(10)
        x[0] = X1[i, j]
        x[1] = X2[i, j]
        
        # Evaluate function
        Z[i, j] = step_func.evaluate(x)

# Create a figure with two subplots
fig = plt.figure(figsize=(15, 6))

# 3D Surface plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8, antialiased=True)
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_zlabel('$f(x)$')
ax1.set_title('Step Function - 3D Surface')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

# Contour plot
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X1, X2, Z, levels=20, cmap='viridis')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_title('Step Function - Contour Plot')
fig.colorbar(contour, ax=ax2)

# Annotate global minimum region
min_x, min_y = -0.25, 0.25
ax2.scatter(min_x, min_y, c='red', s=50, marker='*', 
          label='Global minimum region\n$[-0.5, 0.5)^{10}$')
ax2.legend(loc='upper right')

# Add grid lines at step boundaries
ax2.grid(color='gray', linestyle='--', linewidth=0.5)
ax2.set_xticks(np.arange(-5, 6, 1.0))
ax2.set_yticks(np.arange(-5, 6, 1.0))

plt.suptitle('Step Function - 2D Slice (other dimensions fixed at 0)', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)

plt.show()

# Zoomed view around the global minimum
plt.figure(figsize=(8, 6))
x1_zoom = np.linspace(-1.5, 1.5, 300)
x2_zoom = np.linspace(-1.5, 1.5, 300)
X1_zoom, X2_zoom = np.meshgrid(x1_zoom, x2_zoom)

Z_zoom = np.zeros_like(X1_zoom)
for i in range(X1_zoom.shape[0]):
    for j in range(X1_zoom.shape[1]):
        x = np.zeros(10)
        x[0] = X1_zoom[i, j]
        x[1] = X2_zoom[i, j]
        Z_zoom[i, j] = step_func.evaluate(x)

plt.contourf(X1_zoom, X2_zoom, Z_zoom, levels=10, cmap='viridis')
plt.colorbar(label='$f(x)$')

# Highlight the global minimum region
plt.axhspan(-0.5, 0.5, -0.5, 0.5, alpha=0.3, color='red', label='Global minimum region')

# Add grid lines at step boundaries
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.xticks(np.arange(-1.5, 1.6, 0.5))
plt.yticks(np.arange(-1.5, 1.6, 0.5))

# Mark half-integer boundaries where discontinuities occur
for i in np.arange(-1.5, 2.0, 1.0):
    plt.axvline(i - 0.5, color='white', linestyle='-', linewidth=1.0)
    plt.axhline(i - 0.5, color='white', linestyle='-', linewidth=1.0)

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Step Function - Zoomed View of Global Minimum Region')
plt.legend()
plt.tight_layout()

plt.show()

# Create a 1D slice to better visualize the step discontinuities
plt.figure(figsize=(10, 6))
x_1d = np.linspace(-3, 3, 1000)
y_1d = np.zeros_like(x_1d)

for i, xi in enumerate(x_1d):
    x = np.zeros(10)
    x[0] = xi
    y_1d[i] = step_func.evaluate(x)

plt.plot(x_1d, y_1d, 'b-', linewidth=2)
plt.grid(True)
plt.axhline(y=30, color='r', linestyle='--', label='Minimum value (bias=30)')

# Annotate the step transitions
for i in range(-3, 4):
    x_boundary = i - 0.5
    if i != 0:  # Skip at 0 to not overlap with the global minimum annotation
        plt.axvline(x_boundary, color='gray', linestyle='--')
        plt.text(x_boundary + 0.05, 32, f'x={x_boundary}', rotation=90)

# Highlight the global minimum region
plt.axvspan(-0.5, 0.5, alpha=0.2, color='green', label='Global minimum region')

plt.xlabel('$x_1$ (other dimensions fixed at 0)')
plt.ylabel('$f(x)$')
plt.title('Step Function - 1D Slice showing the Step Discontinuities')
plt.legend()
plt.tight_layout()

plt.show() 