import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot(a, x_test, result, name, width=2):
    # Define the 2D region
    x = np.linspace(a[0] - width, a[0] + width, 100)
    y = np.linspace(a[1] - width, a[1] + width, 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate the result function on the grid
    Z = np.array([[result((X[i, j], Y[i, j])) for j in range(X.shape[1])] for i in range(X.shape[0])])

    # Plot the result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Nested Pade Approximation')

    # Save the plot to a file
    plt.savefig(f'nested_pade_approximation_{name}.png')
