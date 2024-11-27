# # import os
# # import numpy as np
# # import open3d as o3d
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import Dataset, DataLoader

# # # Function to read KITTI Velodyne .bin file
# # def read_kitti_bin(file_path):
# #     """
# #     Read a KITTI Velodyne .bin file and extract 3D point cloud (x, y, z).
# #     :param file_path: Path to the .bin file.
# #     :return: numpy array of shape (N, 3), where N is the number of points.
# #     """
# #     points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
# #     return points[:, :3]  # Extract only x, y, z coordinates

# # # Function to visualize a point cloud using Open3D
# # def visualize_point_cloud(points):
# #     """
# #     Visualize a 3D point cloud using Open3D.
# #     :param points: numpy array of shape (N, 3).
# #     """
# #     pcd = o3d.geometry.PointCloud()
# #     pcd.points = o3d.utility.Vector3dVector(points)
# #     o3d.visualization.draw_geometries([pcd])

# # # Custom Dataset for Point Cloud Data
# # class PointCloudDataset(Dataset):
# #     def __init__(self, point_clouds):
# #         self.point_clouds = point_clouds

# #     def __len__(self):
# #         return len(self.point_clouds)

# #     def __getitem__(self, idx):
# #         return torch.tensor(self.point_clouds[idx], dtype=torch.float32)

# # # Neural Network (Autoencoder) for Point Cloud Data
# # class PointCloudAutoencoder(nn.Module):
# #     def __init__(self):
# #         super(PointCloudAutoencoder, self).__init__()
# #         self.encoder = nn.Sequential(
# #             nn.Linear(3, 64),
# #             nn.ReLU(),
# #             nn.Linear(64, 128),
# #             nn.ReLU()
# #         )
# #         self.decoder = nn.Sequential(
# #             nn.Linear(128, 64),
# #             nn.ReLU(),
# #             nn.Linear(64, 3)  # Reconstruct x, y, z
# #         )

# #     def forward(self, x):
# #         x = self.encoder(x)
# #         x = self.decoder(x)
# #         return x

# # # Main code to load data, train the neural network, and visualize point cloud
# # if __name__ == "__main__":
# #     # Path to the folder containing .bin files
# #     folder_path = "/home/taherk/Downloads/2011_09_28_drive_0035_sync/2011_09_28/2011_09_28_drive_0035_sync/velodyne_points/data"  # Replace with your actual folder path

# #     # Step 1: Read and combine all .bin files into a single dataset
# #     all_points = []
# #     for file_name in sorted(os.listdir(folder_path)):
# #         if file_name.endswith('.bin'):
# #             file_path = os.path.join(folder_path, file_name)
# #             points = read_kitti_bin(file_path)
# #             all_points.append(points)

# #     # Concatenate all point clouds into a single array
# #     all_points = np.concatenate(all_points, axis=0)

# #     # Visualize the combined point cloud
# #     print("Visualizing combined point cloud...")
# #     visualize_point_cloud(all_points)

# #     # Step 2: Load data into DataLoader
# #     dataset = PointCloudDataset(all_points)
# #     dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# #     # Step 3: Initialize Neural Network, Loss, and Optimizer
# #     model = PointCloudAutoencoder()
# #     criterion = nn.MSELoss()
# #     optimizer = optim.Adam(model.parameters(), lr=0.001)

# #     # Step 4: Training Loop
# #     print("Starting training...")
# #     for epoch in range(10):  # Adjust epochs as needed
# #         total_loss = 0
# #         for batch in dataloader:
# #             optimizer.zero_grad()
# #             reconstructed = model(batch)
# #             loss = criterion(reconstructed, batch)
# #             loss.backward()
# #             optimizer.step()
# #             total_loss += loss.item()
# #         print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# #     # Step 5: Save the trained model
# #     torch.save(model.state_dict(), "pointcloud_autoencoder.pth")
# #     print("Model saved as 'pointcloud_autoencoder.pth'.")

# #     # Step 6: Visualize reconstructed point cloud (optional)
# #     print("Visualizing reconstructed point cloud...")
# #     with torch.no_grad():
# #         reconstructed_points = model(torch.tensor(all_points, dtype=torch.float32)).numpy()
# #         visualize_point_cloud(reconstructed_points)

# # # import os
# # # import numpy as np
# # # import open3d as o3d
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # from torch.utils.data import Dataset, DataLoader

# # # def read_kitti_bin(file_path):
# # #     """
# # #     Read a KITTI Velodyne .bin file and extract 3D point cloud (x, y, z).
# # #     :param file_path: Path to the .bin file.
# # #     :return: numpy array of shape (N, 3), where N is the number of points.
# # #     """
# # #     points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
# # #     return points[:, :3]

# # # def visualize_point_cloud(points):
# # #     """
# # #     Visualize a 3D point cloud using Open3D.
# # #     :param points: numpy array of shape (N, 3).
# # #     """
# # #     pcd = o3d.geometry.PointCloud()
# # #     pcd.points = o3d.utility.Vector3dVector(points)
# # #     o3d.visualization.draw_geometries([pcd])

# # # if __name__ == "__main__":
# # #     folder_path = "/home/taherk/Downloads/2011_09_28_drive_0035_sync/2011_09_28/2011_09_28_drive_0035_sync/velodyne_points/data"  # Replace with your actual folder path
# # #     print("Files in folder:", os.listdir(folder_path))


# # #     # Step 1: Read and combine all .bin files into a single dataset
# # #     all_points = []
# # #     for file_name in sorted(os.listdir(folder_path)):
# # #         if file_name.endswith('.bin'):
# # #             file_path = os.path.join(folder_path, file_name)
# # #             print(f"Processing file: {file_path}")
# # #             try:
# # #                 points = read_kitti_bin(file_path)
# # #                 all_points.append(points)
# # #             except Exception as e:
# # #                 print(f"Failed to process {file_path}: {e}")

# # #     if not all_points:
# # #         print("No valid .bin files found in the folder. Please check the folder path and file extensions.")
# # #         exit(1)

# # #     # Concatenate all point clouds into a single array
# # #     all_points = np.concatenate(all_points, axis=0)
# # #     print(f"Combined point cloud shape: {all_points.shape}")

# # #     # Visualize the combined point cloud
# # #     visualize_point_cloud(all_points)


# import os
# import numpy as np
# import open3d as o3d
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import copy
# from datetime import datetime
# import networkx as nx
# import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from models.unet_3d import UNet3D  # Updated to use UNet3D
# from optimizers.dinno import DiNNO
# from utils import graph_generation

# torch.set_default_dtype(torch.double)

# # Function to read KITTI Velodyne .bin file
# def read_kitti_bin(file_path):
#     """
#     Read a KITTI Velodyne .bin file and extract 3D point cloud (x, y, z).
#     :param file_path: Path to the .bin file.
#     :return: numpy array of shape (1, N, 3), where N is the number of points and 1 is the channel dimension.
#     """
#     points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
#     points = points[:, :3]  # Extract only x, y, z coordinates
#     points = np.expand_dims(points, axis=0)  # Adding channel dimension for the model
#     return points

# # Function to visualize a point cloud using Open3D
# def visualize_point_cloud(points):
#     """
#     Visualize a 3D point cloud using Open3D.
#     :param points: numpy array of shape (N, 3).
#     """
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     o3d.visualization.draw_geometries([pcd])

# # Custom Dataset for Point Cloud Data
# class PointCloudDataset(Dataset):
#     def __init__(self, point_clouds):
#         self.point_clouds = point_clouds

#     def __len__(self):
#         return len(self.point_clouds)

#     def __getitem__(self, idx):
#         return torch.tensor(self.point_clouds[idx], dtype=torch.float32)

# def train_dinno(model, loss, train_sets, val_set, graph, device, conf):
#     # Define the DiNNO optimizer
#     optimizer = DiNNO(graph, model, loss, train_sets, val_set, device, conf)
    
#     for epoch in range(conf['epochs']):
#         print(f"Epoch {epoch + 1} started")
#         optimizer.train()
#         if conf['verbose']:
#             print(f"Epoch {epoch + 1} completed")
    
#     return optimizer.metrics

# if __name__ == "__main__":
#     # Configuration
#     conf = {
#         "output_metadir": "./output/",
#         "name": "3d_map_DiNNO",
#         "epochs": 6,
#         "verbose": True,
#         "graph": {
#             "type": "cycle",
#             "num_nodes": 10,
#             "p": 0.3,
#             "gen_attempts": 100
#         },
#         "train_batch_size": 16,  # Reduced to avoid memory issues
#         "val_batch_size": 16,
#         "data_split_type": "random",  # Changed to random since labels are unavailable
#         "data_dir": "/home/taherk/Downloads/2011_09_28_drive_0035_sync/2011_09_28/2011_09_28_drive_0035_sync/velodyne_points/data",
#         "model": {
#             "in_channels": 1,
#             "out_channels": 3,  # Adjusted if output should reconstruct x, y, z
#             "init_features": 3,
#             "kernel_size": 5,
#             "linear_width": 64
#         },
#         "loss": "MSE",  # Changed loss to MSE as NLL requires log_softmax output
#         "use_cuda": torch.cuda.is_available(),
#         "individual_training": {
#             "train_solo": False,
#             "optimizer": "adam",
#             "lr": 0.005,
#             "verbose": True
#         }
#     }

#     # Create output directory
#     if not os.path.exists(conf["output_metadir"]):
#         os.makedirs(conf["output_metadir"], exist_ok=True)

#     # Create communication graph
#     try:
#         N, graph = graph_generation.generate_from_conf(conf["graph"])
#     except NameError as e:
#         print("Error:", str(e))
#         available_graph_types = ["fully_connected", "ring", "star", "erdos_renyi", "cycle"]  # Example available graph types
#         print("Available graph types:", available_graph_types)
#         raise ValueError("Unknown communication graph type. Please check the graph configuration.")
#     nx.write_gpickle(graph, os.path.join(conf["output_metadir"], "graph.gpickle"))

#     # Load point cloud data
#     all_points = []
#     for file_name in sorted(os.listdir(conf["data_dir"])):
#         if file_name.endswith('.bin'):
#             file_path = os.path.join(conf["data_dir"], file_name)
#             points = read_kitti_bin(file_path)
#             all_points.append(points)

#     all_points = np.concatenate(all_points, axis=0)
#     visualize_point_cloud(all_points[:, 0, :])  # Visualize only x, y, z

#     # Split data into subsets for each node
#     num_samples_per = len(all_points) // N
#     point_splits = [num_samples_per for _ in range(N)]
#     train_subsets = torch.utils.data.random_split(all_points, point_splits)

#     val_set = PointCloudDataset(all_points[:1000])  # Example validation set

#     # Create base model
#     model = UNet3D(
#         in_channels=conf["model"]["in_channels"],
#         out_channels=conf["model"]["out_channels"],
#         init_features=conf["model"]["init_features"]
#     )

#     # Define base loss function
#     if conf["loss"] == "MSE":
#         loss = torch.nn.MSELoss()
#     else:
#         raise NameError("Unknown loss function.")

#     # Assign device
#     device = torch.device("cuda" if conf["use_cuda"] else "cpu")
#     print(f"Device is set to {'GPU' if device.type == 'cuda' else 'CPU'}")

#     # Train using DiNNO
#     if conf["individual_training"]["train_solo"]:
#         print("Performing individual training...")
#     else:
#         metrics = train_dinno(copy.deepcopy(model), loss, train_subsets, val_set, graph, device, conf)

#         # Save metrics and model
#         torch.save(metrics, os.path.join(conf["output_metadir"], "dinno_metrics.pt"))
#         torch.save(model.state_dict(), os.path.join(conf["output_metadir"], "dinno_trained_model.pth"))
#         print("Training complete. Metrics and model saved.")

# # ### Summary of Changes:
# # 1. **Data Formatting**:
# #    - Added a channel dimension for the point cloud data to fit into `UNet3D`.

# # 2. **Graph Generation**:
# #    - Changed data splitting to `random` since the label (`classes`) information was unavailable in `.bin` data.

# # 3. **Batch Size**:
# #    - Reduced batch sizes to avoid memory overload.

# # Try these adjustments, and let me know if it still doesn't work or if additional debugging is needed.


import os
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy
from datetime import datetime
import networkx as nx
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_3d import UNet3D  # Updated to use UNet3D
from optimizers.dinno import DiNNO
from utils import graph_generation

torch.set_default_dtype(torch.double)

# Function to read KITTI Velodyne .bin file
def read_kitti_bin(file_path):
    """
    Read a KITTI Velodyne .bin file and extract 3D point cloud (x, y, z).
    :param file_path: Path to the .bin file.
    :return: numpy array of shape (1, N, 3), where N is the number of points and 1 is the channel dimension.
    """
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]  # Extract only x, y, z coordinates
    points = np.expand_dims(points, axis=0)  # Adding channel dimension for the model
    return points

# Function to visualize a point cloud using Open3D
def visualize_point_cloud(points):
    """
    Visualize a 3D point cloud using Open3D.
    :param points: numpy array of shape (N, 3).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

# Custom Dataset for Point Cloud Data
class PointCloudDataset(Dataset):
    def __init__(self, point_clouds):
        self.point_clouds = point_clouds

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        return torch.tensor(self.point_clouds[idx], dtype=torch.float32)

def train_dinno(model, loss, train_sets, val_set, graph, device, conf):
    # Define the DiNNO optimizer
    optimizer = DiNNO(graph, model, loss, train_sets, val_set, device, conf)
    
    for epoch in range(conf['epochs']):
        print(f"Epoch {epoch + 1} started")
        optimizer.train()
        if conf['verbose']:
            print(f"Epoch {epoch + 1} completed")
    
    return optimizer.metrics

if __name__ == "__main__":
    # Configuration
    conf = {
        "output_metadir": "./output/",
        "name": "3d_map_DiNNO",
        "epochs": 6,
        "verbose": True,
        "graph": {
            "type": "cycle",
            "num_nodes": 10,
            "p": 0.3,
            "gen_attempts": 100
        },
        "train_batch_size": 16,  # Reduced batch size to avoid memory issues
        "val_batch_size": 16,
        "data_split_type": "random",  # Changed to random since labels are unavailable
        "data_dir": "/home/taherk/Downloads/2011_09_28_drive_0035_sync/2011_09_28/2011_09_28_drive_0035_sync/velodyne_points/data",
        "model": {
            "in_channels": 1,
            "out_channels": 3,  # Adjusted to match x, y, z reconstruction
            "init_features": 1,  # Reduced to minimize memory consumption
            "kernel_size": 3,
            "linear_width": 64
        },
        "loss": "MSE",  # Changed loss to MSE as NLL requires log_softmax output
        "use_cuda": torch.cuda.is_available(),
        "individual_training": {
            "train_solo": False,
            "optimizer": "adam",
            "lr": 0.005,
            "verbose": True
        }
    }

    # Create output directory
    if not os.path.exists(conf["output_metadir"]):
        os.makedirs(conf["output_metadir"], exist_ok=True)

    # Create communication graph
    try:
        N, graph = graph_generation.generate_from_conf(conf["graph"])
    except NameError as e:
        print("Error:", str(e))
        available_graph_types = ["fully_connected", "ring", "star", "erdos_renyi", "cycle"]  # Example available graph types
        print("Available graph types:", available_graph_types)
        raise ValueError("Unknown communication graph type. Please check the graph configuration.")
    nx.write_gpickle(graph, os.path.join(conf["output_metadir"], "graph.gpickle"))

    # Load point cloud data
    all_points = []
    for file_name in sorted(os.listdir(conf["data_dir"])):
        if file_name.endswith('.bin'):
            file_path = os.path.join(conf["data_dir"], file_name)
            points = read_kitti_bin(file_path)
            all_points.append(points)

    all_points = np.concatenate(all_points, axis=0)
    all_points = all_points[:, 0, :]  # Remove channel dimension for visualization
    visualize_point_cloud(all_points[:1000])  # Visualize only a small portion of the data

    # Split data into subsets for each node
    num_samples_per = len(all_points) // N
    point_splits = [num_samples_per for _ in range(N)]
    train_subsets = torch.utils.data.random_split(all_points, point_splits)

    val_set = PointCloudDataset(all_points[:1000])  # Example validation set

    # Create base model
    model = UNet3D(
        in_channels=conf["model"]["in_channels"],
        out_channels=conf["model"]["out_channels"],
        init_features=conf["model"]["init_features"]
    )

    # Define base loss function
    if conf["loss"] == "MSE":
        loss = torch.nn.MSELoss()
    else:
        raise NameError("Unknown loss function.")

    # Assign device
    device = torch.device("cuda" if conf["use_cuda"] else "cpu")
    print(f"Device is set to {'GPU' if device.type == 'cuda' else 'CPU'}")

    # Train using DiNNO
    if conf["individual_training"]["train_solo"]:
        print("Performing individual training...")
        # Implement individual training logic here if needed
    else:
        metrics = train_dinno(copy.deepcopy(model), loss, train_subsets, val_set, graph, device, conf)

        # Save metrics and model
        torch.save(metrics, os.path.join(conf["output_metadir"], "dinno_metrics.pt"))
        torch.save(model.state_dict(), os.path.join(conf["output_metadir"], "dinno_trained_model.pth"))
        print("Training complete. Metrics and model saved.")
