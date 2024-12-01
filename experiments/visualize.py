# import open3d as o3d
# import os

# def visualize_pcd(pcd_path):
#     """
#     Loads and visualizes a point cloud from a PCD file using Open3D.
    
#     Args:
#         pcd_path (str): Path to the PCD file.
#     """
#     if not os.path.exists(pcd_path):
#         print(f"File not found: {pcd_path}")
#         return
    
#     # Load the point cloud
#     global_map = o3d.io.read_point_cloud(pcd_path)
    
#     if global_map.is_empty():
#         print("The point cloud is empty. Please check the file content.")
#         return
    
#     # Optional: Downsample for better visualization performance
#     voxel_size = 0.05  # Adjust voxel size as needed
#     global_map_downsampled = global_map.voxel_down_sample(voxel_size=voxel_size)
    
#     # Print point cloud information
#     print(global_map_downsampled)
    
#     # Visualize
#     o3d.visualization.draw_geometries(
#         [global_map_downsampled],
#         window_name="Reconstructed Global Map",
#         width=800,
#         height=600,
#         left=50,
#         top=50,
#         point_show_normal=False
#     )

# if __name__ == "__main__":
#     # Path to your reconstructed global map PCD file
#     pcd_file = "./output/reconstructed_global_map.pcd"  # Update this path if different
    
#     visualize_pcd(pcd_file)


import open3d as o3d
import numpy as np
import os

def visualize_reconstructed_map(pcd_path):
    """
    Loads and visualizes a reconstructed point cloud from a .npy file using Open3D.

    Args:
        pcd_path (str): Path to the reconstructed_map.npy file.
    """
    if not os.path.exists(pcd_path):
        print(f"File not found: {pcd_path}")
        return

    # Load the reconstructed map from .npy
    reconstructed_points = np.load(pcd_path)
    print(f"Loaded reconstructed points shape: {reconstructed_points.shape}, dtype: {reconstructed_points.dtype}")

    # Ensure the data is float64
    if reconstructed_points.dtype != np.float64:
        print("Converting reconstructed points to float64...")
        reconstructed_points = reconstructed_points.astype(np.float64)

    # Ensure the shape is (N, 3)
    if reconstructed_points.ndim != 2 or reconstructed_points.shape[1] != 3:
        raise ValueError(f"Reconstructed points have invalid shape: {reconstructed_points.shape}. Expected shape (N, 3).")

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    try:
        pcd.points = o3d.utility.Vector3dVector(reconstructed_points)
    except RuntimeError as e:
        print(f"Error assigning points to PointCloud: {e}")
        return

    # Optional: Downsample for better visualization performance
    voxel_size = 0.05  # Adjust as needed
    pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Downsampled point cloud has {len(pcd_downsampled.points)} points.")

    # Visualize the point cloud
    o3d.visualization.draw_geometries(
        [pcd_downsampled],
        window_name="Reconstructed Global Map",
        width=800,
        height=600,
        left=50,
        top=50,
        point_show_normal=False
    )

if __name__ == "__main__":
    # Path to the reconstructed_map.npy file
    pcd_file = "./output/reconstructed_map.npy"  # Update this path if different

    visualize_reconstructed_map(pcd_file)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
import os

def plot_reconstructed_map(pcd_path):
    """
    Loads and plots a reconstructed point cloud from a .npy file using Matplotlib.

    Args:
        pcd_path (str): Path to the reconstructed_map.npy file.
    """
    if not os.path.exists(pcd_path):
        print(f"File not found: {pcd_path}")
        return

    # Load the reconstructed map from .npy
    reconstructed_points = np.load(pcd_path)
    print(f"Loaded reconstructed points shape: {reconstructed_points.shape}, dtype: {reconstructed_points.dtype}")

    # Ensure the data is float64 for better precision
    if reconstructed_points.dtype != np.float64:
        print("Converting reconstructed points to float64...")
        reconstructed_points = reconstructed_points.astype(np.float64)

    # Ensure the shape is (N, 3)
    if reconstructed_points.ndim != 2 or reconstructed_points.shape[1] != 3:
        raise ValueError(f"Reconstructed points have invalid shape: {reconstructed_points.shape}. Expected shape (N, 3).")

    # Check for NaN or Inf
    if np.isnan(reconstructed_points).any() or np.isinf(reconstructed_points).any():
        print("Warning: Point cloud contains NaN or Inf values. Removing them.")
        reconstructed_points = reconstructed_points[~np.isnan(reconstructed_points).any(axis=1)]
        reconstructed_points = reconstructed_points[~np.isinf(reconstructed_points).any(axis=1)]
        print(f"After removing NaN/Inf, shape: {reconstructed_points.shape}")

    # Optional: Downsample for better visualization performance
    num_points = 10000  # Adjust as needed
    if reconstructed_points.shape[0] > num_points:
        indices = np.random.choice(reconstructed_points.shape[0], num_points, replace=False)
        sampled_points = reconstructed_points[indices]
        print(f"Downsampled to {num_points} points for visualization.")
    else:
        sampled_points = reconstructed_points
        print(f"Using all {sampled_points.shape[0]} points for visualization.")

    # Verify that sampled_points is not empty
    if sampled_points.shape[0] == 0:
        print("No points to plot.")
        return

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    try:
        # Scatter plot without edgecolors to avoid broadcasting issues
        scatter = ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
                             c='blue', s=1, alpha=0.6)  # Removed edgecolors='none'
    except ValueError as e:
        print(f"ValueError during scatter plot: {e}")
        return
    except Exception as e:
        print(f"Unexpected error during scatter plot: {e}")
        return

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set title
    ax.set_title('Reconstructed Global Map')

    # Equal aspect ratio for all axes
    try:
        max_range = np.array([sampled_points[:, 0].max()-sampled_points[:, 0].min(),
                              sampled_points[:, 1].max()-sampled_points[:, 1].min(),
                              sampled_points[:, 2].max()-sampled_points[:, 2].min()]).max() / 2.0

        mid_x = (sampled_points[:, 0].max()+sampled_points[:, 0].min()) * 0.5
        mid_y = (sampled_points[:, 1].max()+sampled_points[:, 1].min()) * 0.5
        mid_z = (sampled_points[:, 2].max()+sampled_points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    except Exception as e:
        print(f"Error setting axis limits: {e}")

    plt.show()

if __name__ == "__main__":
    # Path to the reconstructed_map.npy file
    # pcd_file = "E:/Documents/Masters_Courses/ESE 5460/final_project_env/DL_Project/experiments/output/reconstructed_map.npy"  # Update this path if different
    # pcd_file = "E:/Documents/Masters_Courses/ESE 5460/final_project_env/DL_Project/output/reconstructed_map.npy" 
    # pcd_file = "E:/Documents/Masters_Courses/ESE 5460/final_project_env/DL_Project/output/3d_map_DiNNO_20241129_125817/original_maps/combined_original_map.npy"
    pcd_file = "E:/Documents/Masters_Courses/ESE 5460/final_project_env/DL_Project/output/3d_map_DiNNO_20241129_125817/reconstructed_maps/combined_global_map.npy"
    plot_reconstructed_map(pcd_file)
    
