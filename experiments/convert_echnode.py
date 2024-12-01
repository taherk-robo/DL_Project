import open3d as o3d
import numpy as np
import os

def visualize_individual_nodes(output_dir, num_nodes=10):
    """
    Visualizes reconstructed maps from individual nodes.
    
    Args:
        output_dir (str): Directory containing reconstructed_map.npy or individual node files.
        num_nodes (int): Number of nodes.
    """
    for i in range(num_nodes):
        pcd_file = os.path.join(output_dir, f"reconstructed_map_node_{i}.npy")  # Adjust naming if different
        if not os.path.exists(pcd_file):
            print(f"Reconstructed map for node {i} not found: {pcd_file}")
            continue
        
        # Load the reconstructed map
        reconstructed_points = np.load(pcd_file)
        print(f"Node {i}: Loaded points shape: {reconstructed_points.shape}, dtype: {reconstructed_points.dtype}")
        
        # Ensure correct dtype
        if reconstructed_points.dtype != np.float64:
            reconstructed_points = reconstructed_points.astype(np.float64)
        
        # Create PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(reconstructed_points)
        
        # Visualize
        print(f"Visualizing Node {i} Reconstruction...")
        o3d.visualization.draw_geometries([pcd], window_name=f"Node {i} Reconstructed Map")

if __name__ == "__main__":
    output_directory = "./output/"  # Update if different
    visualize_individual_nodes(output_directory, num_nodes=10)
