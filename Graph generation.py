import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Define a 3D line
# Parametric form: (x, y, z) = (x0, y0, z0) + t * (a, b, c)
x0, y0, z0 = 0, 0, 0  # Line passes through this point
a, b, c = 1, 1, 1  # Direction vector of the line

# Generate points along the line
t_values = np.linspace(-10, 10, 100)  # Adjust the range as needed
line_x = x0 + t_values * a
line_y = y0 + t_values * b
line_z = z0 + t_values * c

# Step 2: Generate random points around the line
# Control the spread of points around the line
num_points = 200
spread = 0.5  # Adjust for wider spread

random_t = np.random.uniform(-3, 3, num_points)  # Random t-values
random_offsets = np.random.normal(0, spread, (num_points, 3))  # Random offsets

# Generate random points
points_x = x0 + random_t * a + random_offsets[:, 0]
points_y = y0 + random_t * b + random_offsets[:, 1]
points_z = z0 + random_t * c + random_offsets[:, 2]

# Step 3: Plot the line and points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the line
ax.plot(line_x, line_y, line_z, color='red', label='Line')

# Plot the random points
ax.scatter(points_x, points_y, points_z, color='blue', alpha=0.6, label='Random Points')

# Set labels and legend
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()

# Display the plot
plt.show()
