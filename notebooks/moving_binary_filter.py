# Johan Hedenström

from scipy.ndimage import generic_filter
import numpy as np

### --------------------------------------------------------------------- ###
# [Example usage]
# from moving_binary_filter import filter
# energy = np.array(outLists[0])
# tof = np.array(outLists[1])
# data = np.vstack((energy, tof)).T
# filtered_data = filter(data, 5, 5)
# x = filtered_data[:, 0]
# y = filtered_data[:, 1]
# 
# [Function arguments]
# data - a 2D array on format (n, 2)
# filter_size - size of the NxN filter
# filter_treshold - maximum neighbours in filter to classify center as noise  
### --------------------------------------------------------------------- ###
def filter(data, filter_size, filter_treshold):
    def noise_removal_filter(values):
        center_idx = len(values) // 2
        center = values[center_idx]
        neighbor_count = np.sum(values) - center
        if neighbor_count <= filter_treshold:
            return 0
        return center

    def generate_grid(data):
        # Grid boundaries
        max_x = np.max(data[:, 0])
        max_y = np.max(data[:, 1])

        # Initialize 2D grids with zeros
        binary_grid = np.zeros((max_x + 1, max_y + 1), dtype=int)
        density_grid = np.zeros((max_x + 1, max_y + 1), dtype=int)

        # Populate grids
        for x, y in data:
            binary_grid[int(x), int(y)] = 1
            density_grid[int(x), int(y)] += 1

        return binary_grid, density_grid

    # Generate grids
    binary_grid, density_grid = generate_grid(data)

    # Filter binary grid
    filtered_binary_grid = generic_filter(binary_grid, noise_removal_filter, size=filter_size)

    # Convert back to data matrix
    X = []
    Y = []
    
    h, w = filtered_binary_grid.shape
    for x in range(h):
        for y in range(w):
            if filtered_binary_grid[x, y] == 1:
                for _ in range(density_grid[x, y]):
                    X.append(x)
                    Y.append(y)

    # Return result with X values in first column and Y in second
    return np.vstack((X, Y)).T