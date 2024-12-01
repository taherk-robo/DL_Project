import numpy as np
import open3d as o3d

# Generate some random points
points = np.random.rand(1000, 3)

# Create Open3D PointCloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Visualize
o3d.visualization.draw_geometries([pcd], window_name="Open3D Test")
