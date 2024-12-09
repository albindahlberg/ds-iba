from scipy.optimize import curve_fit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

# Function describing trajectory curves
def fx(x, a, b, c):
    return -abs(a) / np.sqrt(abs(x + b)) + c

# Approximate perpendicular distance from point to curve
def approximate_distance(x, y, a, b, c):
    y_curve = fx(x, a, b, c)
    dfx = abs(a) * (x + b) / (2 * abs(x + b) ** (5/2))
    d = np.abs(y - y_curve) / np.sqrt(1 + dfx**2)
    return d

# Fit a curve to trajectory points
# x_traj, y_traj - important trajectory points
# x_data, y_data - ALL trajectory cluster points
def fit(x_traj, y_traj, x_data, y_data):  
    # - can't touch this ----------------------------------
    param_space = ([0, 0, -np.inf], [np.inf, 50, np.inf])
    initial_guess = [1, 1, 1]
    # -----------------------------------------------------
    
    # Fit curve to trajectory points
    params, covariance = curve_fit(fx, 
                                   x_traj, 
                                   y_traj,
                                   bounds=param_space,
                                   p0=initial_guess)

    # Compute y-values along the curve
    x_fit = np.linspace(min(x_data), max(x_data), 500)
    y_fit = fx(x_fit, *params)

    # Calculate R2
    y_pred = fx(x_traj, *params) 
    residuals = y_traj - y_pred
    RSS = np.sum(residuals**2)
    TSS = np.sum((y_traj - np.mean(y_traj))**2)
    R2 = 1 - (RSS / TSS)

    # Check if there was an issue with the curve fit
    # (we expect the curve to fit almost perfectly 
    # in case all points belong to the same trajectory)
    if R2 < 0.99:         
        # Compute approximate distances from each point to the curve
        approximate_distances = approximate_distance(x_data, y_data, *params)

        # Create a weighted histogram, indicating the overall density along the curve
        # (in case some points belong to another trajectory, and the trajectories are
        # well separated, we expect a density gap somewhere in the beginning)
        density, bin_edges = np.histogram(x_data, bins=50, weights=1 / (1 + approximate_distances))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find the density gap (lowest density) in the first half of the histogram
        valley_index = np.argmin(density[:len(density) // 2])
        valley_x = bin_centers[valley_index]

        # Filter out the potentially incorrect trajectory points
        valid_indices = x_traj >= valley_x
        x_traj_filtered = x_traj[valid_indices]
        y_traj_filtered = y_traj[valid_indices]

        # Re-fit curve to the filtered data
        params, covariance = curve_fit(fx, 
                                       x_traj_filtered, 
                                       y_traj_filtered, 
                                       bounds=param_space,
                                       p0=initial_guess)

    return params

# Function to find trajectory boundary
def find_domain(x_data, y_data, params, radius=10, density_threshold=3):
    xx = np.linspace(min(x_data), max(x_data), 500)
    yy = fx(xx, *params)

    density = []
    for i in range(len(xx)):
        x, y = xx[i], yy[i]  

        # Compute distance to all points from trajectory for each x
        distances = np.sqrt((x_data - x)**2 + (y_data - y)**2)

        # Count points within radius ("density")
        density_value = np.sum(distances <= radius)
        density.append(density_value)

    # Find where the trajectory likely ends
    non_zero_indices = np.where(np.array(density) > density_threshold)[0]
    x_end = xx[non_zero_indices[-1]] if non_zero_indices[-1] < len(xx) - 1 else max(x_data)

    return min(x_data), x_end

# Function to find confidence bands
def compute_confidence_band(x_data, y_data, params):
    pass
