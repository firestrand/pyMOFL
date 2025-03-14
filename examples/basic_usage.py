"""
Basic usage examples for the pyMOFL library.

This script demonstrates how to use the basic functions, transformations, and compositions
provided by the pyMOFL library.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import the pyMOFL library
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pyMOFL.functions.unimodal import SphereFunction, RosenbrockFunction
from src.pyMOFL.functions.multimodal import RastriginFunction
from src.pyMOFL.decorators import ShiftedFunction, RotatedFunction
from src.pyMOFL.composites import CompositeFunction


def plot_2d_function(func, bounds, title, resolution=100):
    """
    Plot a 2D function as a surface.
    
    Args:
        func: The function to plot.
        bounds: The bounds of the function.
        title: The title of the plot.
        resolution: The resolution of the plot.
    """
    x = np.linspace(bounds[0, 0], bounds[0, 1], resolution)
    y = np.linspace(bounds[1, 0], bounds[1, 1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Evaluate the function at each point
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func.evaluate(np.array([X[i, j], Y[i, j]]))
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')
    ax.set_title(title)
    
    plt.show()


def main():
    """
    Main function to demonstrate the usage of the pyMOFL library.
    """
    # Create a 2D Sphere function
    sphere = SphereFunction(dimension=2)
    print(f"Sphere function at [0, 0]: {sphere.evaluate(np.array([0, 0]))}")
    print(f"Sphere function at [1, 1]: {sphere.evaluate(np.array([1, 1]))}")
    
    # Create a 2D Rastrigin function
    rastrigin = RastriginFunction(dimension=2)
    print(f"Rastrigin function at [0, 0]: {rastrigin.evaluate(np.array([0, 0]))}")
    print(f"Rastrigin function at [1, 1]: {rastrigin.evaluate(np.array([1, 1]))}")
    
    # Create a 2D Rosenbrock function
    rosenbrock = RosenbrockFunction(dimension=2)
    print(f"Rosenbrock function at [1, 1]: {rosenbrock.evaluate(np.array([1, 1]))}")
    print(f"Rosenbrock function at [0, 0]: {rosenbrock.evaluate(np.array([0, 0]))}")
    
    # Create a shifted Sphere function
    shift = np.array([1.0, 2.0])
    shifted_sphere = ShiftedFunction(sphere, shift)
    print(f"Shifted Sphere function at [0, 0]: {shifted_sphere.evaluate(np.array([0, 0]))}")
    print(f"Shifted Sphere function at [1, 2]: {shifted_sphere.evaluate(np.array([1, 2]))}")
    
    # Create a rotated Rastrigin function
    rotation_matrix = np.array([[0.866, -0.5], [0.5, 0.866]])  # 30-degree rotation
    rotated_rastrigin = RotatedFunction(rastrigin, rotation_matrix)
    print(f"Rotated Rastrigin function at [0, 0]: {rotated_rastrigin.evaluate(np.array([0, 0]))}")
    
    # Create a composite function
    components = [sphere, rastrigin, rosenbrock]
    sigmas = [1.0, 2.0, 3.0]
    lambdas = [1.0, 1.0, 1.0]
    biases = [0.0, 100.0, 200.0]
    composite = CompositeFunction(components, sigmas, lambdas, biases)
    print(f"Composite function at [0, 0]: {composite.evaluate(np.array([0, 0]))}")
    print(f"Composite function at [1, 1]: {composite.evaluate(np.array([1, 1]))}")
    
    # Plot the functions
    plot_2d_function(sphere, sphere.bounds, "Sphere Function")
    plot_2d_function(rastrigin, rastrigin.bounds, "Rastrigin Function")
    plot_2d_function(rosenbrock, rosenbrock.bounds, "Rosenbrock Function")
    plot_2d_function(shifted_sphere, shifted_sphere.bounds, "Shifted Sphere Function")
    plot_2d_function(rotated_rastrigin, rotated_rastrigin.bounds, "Rotated Rastrigin Function")
    plot_2d_function(composite, composite.bounds, "Composite Function")


if __name__ == "__main__":
    main() 