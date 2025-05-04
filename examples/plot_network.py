"""
Visualization of the Network function (SPSO ID 11).

This example creates visualizations of the Network function to illustrate
how the BSC positions affect the total network distance and how different
connection patterns impact the objective function.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyMOFL.functions.hybrid import NetworkFunction

# Create the Network function
network = NetworkFunction()

# -------------------- Visualize BTS positions and optimal BSC placement --------------------
plt.figure(figsize=(10, 8))

# Plot BTS positions
plt.scatter(network.bts_positions[:, 0], network.bts_positions[:, 1], 
            c='blue', s=100, marker='s', label='BTS')

# Add BTS indices for reference
for i, (x, y) in enumerate(network.bts_positions):
    plt.text(x + 0.3, y + 0.3, str(i), fontsize=9)

plt.title('Network Problem - BTS Positions')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.grid(True)
plt.legend()
plt.xlim(0, 22)
plt.ylim(0, 18)

plt.tight_layout()
plt.show()

# -------------------- Visualize effect of BSC position --------------------
# Fix a valid connection configuration where:
# - BTS 0-9 connect to BSC 0
# - BTS 10-18 connect to BSC 1
# Vary the position of BSC 0, keeping BSC 1 fixed

# Create a fixed valid solution
x = np.zeros(42)

# Set connections: BTS 0-9 to BSC 0, BTS 10-18 to BSC 1
x[:10] = 1.0         # BTS 0-9 to BSC 0
x[19+10:19+19] = 1.0 # BTS 10-18 to BSC 1

# Fix BSC 1 position
x[40:42] = [15, 10]  # BSC 1 fixed at (15, 10)

# Create a grid of positions for BSC 0
resolution = 50
x_grid = np.linspace(0, 20, resolution)
y_grid = np.linspace(0, 18, resolution)
X, Y = np.meshgrid(x_grid, y_grid)

# Evaluate function for each BSC 0 position
Z = np.zeros_like(X)
for i in range(resolution):
    for j in range(resolution):
        # Set BSC 0 position
        x_copy = x.copy()
        x_copy[38:40] = [X[i, j], Y[i, j]]
        
        # Evaluate
        Z[i, j] = network.evaluate(x_copy)

# Create a figure with contour plot
plt.figure(figsize=(12, 9))

# Contour plot of objective function based on BSC 0 position
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='Objective value')

# Add BTS positions
plt.scatter(network.bts_positions[:10, 0], network.bts_positions[:10, 1], 
            c='blue', s=100, marker='s', label='BTS connected to BSC 0')
plt.scatter(network.bts_positions[10:, 0], network.bts_positions[10:, 1], 
            c='green', s=100, marker='s', label='BTS connected to BSC 1')

# Add BSC 1 position (fixed)
plt.scatter(15, 10, c='red', s=200, marker='^', label='BSC 1 (fixed)')

# Find and mark the optimal BSC 0 position
min_idx = np.unravel_index(np.argmin(Z), Z.shape)
optimal_x, optimal_y = X[min_idx], Y[min_idx]
plt.scatter(optimal_x, optimal_y, c='purple', s=200, marker='^', label=f'Optimal BSC 0: ({optimal_x:.1f}, {optimal_y:.1f})')

plt.title('Network Function - Effect of BSC 0 Position')
plt.xlabel('BSC 0 X-coordinate')
plt.ylabel('BSC 0 Y-coordinate')
plt.grid(True)
plt.legend(loc='upper right')
plt.xlim(0, 20)
plt.ylim(0, 18)

plt.tight_layout()
plt.show()

# -------------------- Visualize effect of different connection patterns --------------------
plt.figure(figsize=(10, 8))

# Fixed BSC positions
bsc0_pos = [5, 5]
bsc1_pos = [15, 10]

# Generate different connection patterns and evaluate
num_patterns = 10
pattern_names = []
values = []

# Create and evaluate different patterns
for i in range(num_patterns):
    # Create a random connection pattern
    if i == 0:
        # Special case: optimal split based on distance
        distances_to_bsc0 = np.array([
            np.sqrt((network.bts_positions[j, 0] - bsc0_pos[0])**2 + 
                    (network.bts_positions[j, 1] - bsc0_pos[1])**2)
            for j in range(network.bts_count)
        ])
        
        distances_to_bsc1 = np.array([
            np.sqrt((network.bts_positions[j, 0] - bsc1_pos[0])**2 + 
                    (network.bts_positions[j, 1] - bsc1_pos[1])**2)
            for j in range(network.bts_count)
        ])
        
        # BTS connects to the closer BSC
        connections = np.zeros(network.bts_count * network.bsc_count)
        for j in range(network.bts_count):
            if distances_to_bsc0[j] <= distances_to_bsc1[j]:
                connections[j] = 1.0  # Connect to BSC 0
            else:
                connections[j + network.bts_count] = 1.0  # Connect to BSC 1
        
        pattern_names.append("Optimal (by distance)")
    
    elif i == 1:
        # All BTS connect to BSC 0
        connections = np.zeros(network.bts_count * network.bsc_count)
        connections[:network.bts_count] = 1.0
        pattern_names.append("All to BSC 0")
    
    elif i == 2:
        # All BTS connect to BSC 1
        connections = np.zeros(network.bts_count * network.bsc_count)
        connections[network.bts_count:2*network.bts_count] = 1.0
        pattern_names.append("All to BSC 1")
    
    else:
        # Random connections - each BTS connects to exactly one BSC
        connections = np.zeros(network.bts_count * network.bsc_count)
        for j in range(network.bts_count):
            bsc_idx = np.random.randint(0, network.bsc_count)
            connections[j + bsc_idx * network.bts_count] = 1.0
        pattern_names.append(f"Random {i-2}")
    
    # Create full solution vector
    x = np.zeros(42)
    x[:network.bts_count * network.bsc_count] = connections
    x[38:40] = bsc0_pos
    x[40:42] = bsc1_pos
    
    # Evaluate
    values.append(network.evaluate(x))

# Plot the results
plt.bar(pattern_names, values, color='teal')
plt.axhline(values[0], color='red', linestyle='--', label=f'Optimal value: {values[0]:.2f}')
plt.title('Network Function - Effect of Different Connection Patterns')
plt.xlabel('Connection Pattern')
plt.ylabel('Objective Value')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -------------------- Visualize a complete solution --------------------
plt.figure(figsize=(12, 10))

# Use the optimal connection pattern from above
x_optimal = np.zeros(42)
x_optimal[:network.bts_count * network.bsc_count] = connections
x_optimal[38:40] = bsc0_pos
x_optimal[40:42] = bsc1_pos

# Extract connections
connections_bsc0 = x_optimal[:network.bts_count] >= 0.5
connections_bsc1 = x_optimal[network.bts_count:2*network.bts_count] >= 0.5

# Plot BTS positions with color indicating which BSC they connect to
plt.scatter(network.bts_positions[connections_bsc0, 0], 
            network.bts_positions[connections_bsc0, 1],
            c='blue', s=100, marker='s', label='BTS connected to BSC 0')

plt.scatter(network.bts_positions[connections_bsc1, 0], 
            network.bts_positions[connections_bsc1, 1],
            c='green', s=100, marker='s', label='BTS connected to BSC 1')

# Plot BSC positions
plt.scatter(bsc0_pos[0], bsc0_pos[1], c='blue', s=200, marker='^', label='BSC 0')
plt.scatter(bsc1_pos[0], bsc1_pos[1], c='green', s=200, marker='^', label='BSC 1')

# Draw connections as lines
for i, is_connected in enumerate(connections_bsc0):
    if is_connected:
        plt.plot([network.bts_positions[i, 0], bsc0_pos[0]], 
                 [network.bts_positions[i, 1], bsc0_pos[1]], 
                 'b-', alpha=0.5)

for i, is_connected in enumerate(connections_bsc1):
    if is_connected:
        plt.plot([network.bts_positions[i, 0], bsc1_pos[0]], 
                 [network.bts_positions[i, 1], bsc1_pos[1]], 
                 'g-', alpha=0.5)

# Add BTS indices
for i, (x, y) in enumerate(network.bts_positions):
    plt.text(x + 0.3, y + 0.3, str(i), fontsize=9)

plt.title(f'Network Function - Optimal Solution (Value: {network.evaluate(x_optimal):.2f})')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.grid(True)
plt.legend()
plt.xlim(0, 22)
plt.ylim(0, 18)

plt.tight_layout()
plt.show() 