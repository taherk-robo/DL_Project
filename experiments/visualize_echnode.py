import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
import os

def plot_global_map(pcd_path):
    """
    Loads and plots a reconstructed global map from a .npy file using Matplotlib.

    Args:
        pcd_path (str): Path to the reconstructed_map.npy file.
    """
    print(f"Attempting to load point cloud from: {pcd_path}")

    if not os.path.exists(pcd_path):
        print(f"Error: File not found: {pcd_path}")
        return

    # Load the reconstructed map from .npy
    try:
        reconstructed_points = np.load(pcd_path)
        print(f"Successfully loaded point cloud.")
    except Exception as e:
        print(f"Error loading .npy file: {e}")
        return

    print(f"Loaded reconstructed points shape: {reconstructed_points.shape}, dtype: {reconstructed_points.dtype}")

    # Ensure the data is float64 for better precision
    if reconstructed_points.dtype != np.float64:
        print("Converting reconstructed points to float64...")
        reconstructed_points = reconstructed_points.astype(np.float64)
        print(f"Data type after conversion: {reconstructed_points.dtype}")

    # Ensure the shape is (N, 3)
    if reconstructed_points.ndim != 2 or reconstructed_points.shape[1] != 3:
        print(f"Error: Reconstructed points have invalid shape: {reconstructed_points.shape}. Expected shape (N, 3).")
        return
    else:
        print(f"Point cloud has a valid shape: {reconstructed_points.shape}")

    # Check for NaN or Inf values
    nan_count = np.isnan(reconstructed_points).sum()
    inf_count = np.isinf(reconstructed_points).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"Warning: Point cloud contains {nan_count} NaN and {inf_count} Inf values. Removing them.")
        reconstructed_points = reconstructed_points[~np.isnan(reconstructed_points).any(axis=1)]
        reconstructed_points = reconstructed_points[~np.isinf(reconstructed_points).any(axis=1)]
        print(f"After removing NaN/Inf, shape: {reconstructed_points.shape}")
    else:
        print("No NaN or Inf values found in the point cloud.")

    # Optional: Downsample for better visualization performance
    num_points = 10000  # Adjust as needed
    total_points = reconstructed_points.shape[0]
    if total_points > num_points:
        print(f"Downsampling from {total_points} to {num_points} points for visualization...")
        indices = np.random.choice(total_points, num_points, replace=False)
        sampled_points = reconstructed_points[indices]
        print(f"Downsampled to {sampled_points.shape[0]} points.")
    else:
        sampled_points = reconstructed_points
        print(f"Using all {sampled_points.shape[0]} points for visualization.")

    # Verify that sampled_points is not empty
    if sampled_points.shape[0] == 0:
        print("Error: No points available to plot after cleaning.")
        return

    # Create a 3D plot
    print("Creating 3D scatter plot...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    try:
        # Scatter plot without edgecolors to avoid broadcasting issues
        scatter = ax.scatter(
            sampled_points[:, 0],
            sampled_points[:, 1],
            sampled_points[:, 2],
            c='blue',
            s=1,
            alpha=0.6
        )
        print("Scatter plot created successfully.")
    except ValueError as e:
        print(f"ValueError during scatter plot: {e}")
        return
    except Exception as e:
        print(f"Unexpected error during scatter plot: {e}")
        return

    # Set labels
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)

    # Set title
    ax.set_title('Reconstructed Global Map', fontsize=15)

    # Equal aspect ratio for all axes
    try:
        print("Setting equal aspect ratio for all axes...")
        max_range = np.array([
            sampled_points[:, 0].max() - sampled_points[:, 0].min(),
            sampled_points[:, 1].max() - sampled_points[:, 1].min(),
            sampled_points[:, 2].max() - sampled_points[:, 2].min()
        ]).max() / 2.0

        mid_x = (sampled_points[:, 0].max() + sampled_points[:, 0].min()) * 0.5
        mid_y = (sampled_points[:, 1].max() + sampled_points[:, 1].min()) * 0.5
        mid_z = (sampled_points[:, 2].max() + sampled_points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        print("Aspect ratio set successfully.")
    except Exception as e:
        print(f"Error setting axis limits: {e}")

    print("Displaying the plot...")
    plt.show()
    print("Plot displayed successfully.")

if __name__ == "__main__":
    # Path to your reconstructed global map .npy file
    pcd_file = "./output/reconstructed_map.npy"  # Update this path if different

    plot_global_map(pcd_file)
