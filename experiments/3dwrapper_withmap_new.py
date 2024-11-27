# import os
# import numpy as np
# import open3d as o3d
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import copy
# from datetime import datetime
# import networkx as nx
# import sys
# import math
# import matplotlib.pyplot as plt
# from pytorch3d.loss import chamfer_distance

# # Adjust the import paths as needed
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from optimizers.dinno import DiNNO
# from utils import graph_generation

# # Function to read KITTI Velodyne .bin file
# def read_kitti_bin(file_path):
#     """
#     Read a KITTI Velodyne .bin file and extract 3D point cloud (x, y, z).
#     :param file_path: Path to the .bin file.
#     :return: numpy array of shape (N, 3), where N is the number of points.
#     """
#     points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
#     return points[:, :3]  # Extract only x, y, z coordinates

# # Function to visualize a point cloud using Open3D
# def visualize_point_cloud(points, title="Point Cloud"):
#     """
#     Visualize a 3D point cloud using Open3D.
#     :param points: numpy array of shape (N, 3).
#     :param title: Title of the visualization window.
#     """
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     o3d.visualization.draw_geometries([pcd], window_name=title)

# # Custom Dataset for Point Cloud Data
# class PointCloudDataset(Dataset):
#     def __init__(self, point_clouds):
#         self.point_clouds = point_clouds
#         # Normalize the point cloud
#         self.point_clouds = (self.point_clouds - np.mean(self.point_clouds, axis=0)) / np.std(self.point_clouds, axis=0)

#     def __len__(self):
#         return len(self.point_clouds)

#     def __getitem__(self, idx):
#         data = torch.tensor(self.point_clouds[idx], dtype=torch.float32)
#         return data, data  # Return a tuple (input, target)

# # Neural Network (PointNet Autoencoder) for Point Cloud Data
# class PointNetAutoencoder(nn.Module):
#     def __init__(self):
#         super(PointNetAutoencoder, self).__init__()
#         # Encoder
#         self.conv1 = nn.Conv1d(3, 64, 1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.conv2 = nn.Conv1d(64, 128, 1)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.conv3 = nn.Conv1d(128, 1024, 1)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.fc1 = nn.Linear(1024, 512)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn5 = nn.BatchNorm1d(256)
#         # Decoder
#         self.fc3 = nn.Linear(256, 512)
#         self.bn6 = nn.BatchNorm1d(512)
#         self.fc4 = nn.Linear(512, 1024)
#         self.bn7 = nn.BatchNorm1d(1024)
#         self.conv4 = nn.Conv1d(1024, 128, 1)
#         self.bn8 = nn.BatchNorm1d(128)
#         self.conv5 = nn.Conv1d(128, 64, 1)
#         self.bn9 = nn.BatchNorm1d(64)
#         self.conv6 = nn.Conv1d(64, 3, 1)

#     def forward(self, x):
#         # Encoder
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)
#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
#         # Decoder
#         x = F.relu(self.bn6(self.fc3(x)))
#         x = F.relu(self.bn7(self.fc4(x)))
#         x = x.view(-1, 1024, 1)
#         x = F.relu(self.bn8(self.conv4(x)))
#         x = F.relu(self.bn9(self.conv5(x)))
#         x = self.conv6(x)
#         x = x.permute(0, 2, 1).contiguous()
#         return x

# # Wrapper class for DDLProblem
# class DDLProblem:
#     def __init__(self, models, N, conf, train_subsets, val_set):
#         self.models = models
#         self.N = N
#         self.conf = conf
#         # Calculate the total number of parameters in the first model
#         self.n = torch.numel(torch.nn.utils.parameters_to_vector(self.models[0].parameters()))
#         # Initialize communication graph
#         self.graph = graph_generation.generate_from_conf(conf["graph"])[1]
#         # Initialize data loaders for each node
#         self.train_loaders = [
#             DataLoader(train_subsets[i], batch_size=conf["train_batch_size"], shuffle=True)
#             for i in range(N)
#         ]
#         self.val_loader = DataLoader(val_set, batch_size=conf["val_batch_size"], shuffle=False)
#         # Define the loss function
#         self.loss_fn = torch.nn.MSELoss()  # Placeholder; will use Chamfer Distance instead
#         # Assign device
#         self.device = self.conf["device"]

#     def local_batch_loss(self, i):
#         """
#         Compute the local batch loss for node i using Chamfer Distance.
#         """
#         model = self.models[i].to(self.device)
#         model.train()
#         try:
#             data, target = next(iter(self.train_loaders[i]))
#         except StopIteration:
#             # Restart the loader if the iterator is exhausted
#             self.train_loaders[i] = iter(self.train_loaders[i])
#             data, target = next(iter(self.train_loaders[i]))
#         data, target = data.to(self.device), target.to(self.device)
#         output = model(data)
#         loss_cd, _ = chamfer_distance(output, target)
#         return loss_cd

#     def evaluate_metrics(self, at_end=False):
#         """
#         Evaluate and return metrics such as validation loss using Chamfer Distance.
#         """
#         metrics = {}
#         for i, model in enumerate(self.models):
#             model.eval()
#             total_loss = 0.0
#             with torch.no_grad():
#                 for data, target in self.val_loader:
#                     data, target = data.to(self.device), target.to(self.device)
#                     output = model(data)
#                     loss_cd, _ = chamfer_distance(output, target)
#                     total_loss += loss_cd.item()
#             average_loss = total_loss / len(self.val_loader)
#             metrics[f'validation_loss_node_{i}'] = average_loss
#             print(f"Validation Loss for node {i}: {average_loss}")
#         return metrics

#     def update_graph(self):
#         """
#         Update the communication graph if needed.
#         """
#         # Implement any dynamic graph updates if required
#         print("Updating communication graph...")
#         # Example: No dynamic updates; keep the graph static
#         pass

# def train_dinno(ddl_problem, loss, val_set, graph, device, conf):
#     # Define the DiNNO optimizer
#     optimizer = DiNNO(ddl_problem, device, conf)
    
#     # Start training
#     optimizer.train()
    
#     return optimizer.metrics

# def align_point_clouds(source_pcd, target_pcd, threshold=0.02):
#     """
#     Align source_pcd to target_pcd using ICP.
#     """
#     transformation = o3d.pipelines.registration.registration_icp(
#         source_pcd, target_pcd, threshold, np.identity(4),
#         o3d.pipelines.registration.TransformationEstimationPointToPoint()
#     ).transformation
#     source_pcd.transform(transformation)
#     return source_pcd

# def reconstruct_and_align_map(ddl_problem, device):
#     """
#     Reconstruct the entire map by aggregating and aligning local reconstructions from all nodes.
#     """
#     reconstructed_pcds = []
#     for i in range(ddl_problem.N):
#         model = ddl_problem.models[i].to(device)
#         model.eval()
#         all_reconstructions = []
#         with torch.no_grad():
#             for data, _ in ddl_problem.train_loaders[i]:
#                 data = data.to(device)
#                 reconstructed = model(data)
#                 all_reconstructions.append(reconstructed.cpu().numpy())
#         reconstructed_points = np.concatenate(all_reconstructions, axis=0)
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(reconstructed_points)
#         pcd = pcd.voxel_down_sample(voxel_size=0.05)  # Optional: Downsample for efficiency
#         reconstructed_pcds.append(pcd)
    
#     # Initialize global map with the first node's reconstruction
#     global_map = reconstructed_pcds[0]
    
#     for pcd in reconstructed_pcds[1:]:
#         global_map = align_point_clouds(pcd, global_map)
#         global_map += pcd
#         global_map = global_map.voxel_down_sample(voxel_size=0.05)  # Optional: Downsample after merging
    
#     return global_map

# if __name__ == "__main__":
#     # Configuration
#     conf = {
#         "output_metadir": "./output/",
#         "name": "3d_map_DiNNO",
#         "epochs": 100,  # Increased number of outer iterations
#         "verbose": True,
#         "graph": {
#             "type": "cycle",
#             "num_nodes": 10,
#             "p": 0.3,
#             "gen_attempts": 100
#         },
#         "train_batch_size": 16,
#         "val_batch_size": 16,
#         "data_split_type": "random",
#         "data_dir": "/home/taherk/Downloads/2011_09_28_drive_0035_sync/2011_09_28/2011_09_28_drive_0035_sync/velodyne_points/data",
#         "model": {
#             "in_channels": 1,
#             "out_channels": 3,
#             "init_features": 3,
#             "kernel_size": 5,
#             "linear_width": 64
#         },
#         "loss": "Chamfer",
#         "use_cuda": torch.cuda.is_available(),
#         "individual_training": {
#             "train_solo": False,
#             "optimizer": "adam",
#             "lr": 0.001,  # Reduced learning rate
#             "verbose": True
#         },
#         # DiNNO Specific Hyperparameters
#         "rho_init": 0.1,               # Reduced initial rho value
#         "rho_scaling": 1.1,            # Scaling factor for rho
#         "lr_decay_type": "constant",   # 'constant', 'linear', or 'log'
#         "primal_lr_start": 0.001,      # Starting learning rate for primal optimizer
#         "primal_lr_finish": 0.0001,    # Final learning rate (used if lr_decay_type is 'linear' or 'log')
#         "outer_iterations": 100,       # Number of outer iterations (set to 'epochs' value)
#         "primal_iterations": 10,       # Number of primal updates per outer iteration
#         "persistant_primal_opt": True,  # Use persistent primal optimizers
#         "primal_optimizer": "adam",     # Type of primal optimizer: 'adam', 'sgd', 'adamw'
#         "metrics_config": {             # Metrics configuration (if used)
#             "evaluate_frequency": 1     # Evaluate metrics every iteration
#         },
#         "device": "cuda" if torch.cuda.is_available() else "cpu"  # Adding 'device' key
#     }

#     # Create output directory
#     if not os.path.exists(conf["output_metadir"]):
#         os.makedirs(conf["output_metadir"], exist_ok=True)

#     # Load point cloud data
#     all_points = []
#     for file_name in sorted(os.listdir(conf["data_dir"])):
#         if file_name.endswith('.bin'):
#             file_path = os.path.join(conf["data_dir"], file_name)
#             points = read_kitti_bin(file_path)
#             all_points.append(points)

#     all_points = np.concatenate(all_points, axis=0)
#     visualize_point_cloud(all_points, title="Original Point Cloud")

#     # Create communication graph
#     try:
#         N, graph = graph_generation.generate_from_conf(conf["graph"])
#     except NameError as e:
#         print("Error:", str(e))
#         available_graph_types = ["fully_connected", "ring", "star", "erdos_renyi", "cycle"]  # Example available graph types
#         print("Available graph types:", available_graph_types)
#         raise ValueError("Unknown communication graph type. Please check the graph configuration.")
#     nx.write_gpickle(graph, os.path.join(conf["output_metadir"], "graph.gpickle"))

#     # Create full training dataset
#     full_train_set = PointCloudDataset(all_points)

#     # Split data into subsets for each node
#     if conf["data_split_type"] == "random":
#         num_samples_per = len(full_train_set) // N
#         point_splits = [num_samples_per for _ in range(N - 1)]
#         point_splits.append(len(full_train_set) - sum(point_splits))  # Ensure the split lengths sum to the total length
#         train_subsets = torch.utils.data.random_split(full_train_set, point_splits)
#     elif conf["data_split_type"] == "hetero":
#         # Assuming the last column is class label; adjust if different
#         classes = torch.unique(torch.tensor(all_points[:, -1]))
#         train_subsets = []
#         if N <= len(classes):
#             joint_labels = torch.tensor(all_points[:, -1])
#             node_classes = torch.split(classes, int(math.ceil(len(classes) / N)))
#             for i in range(N):
#                 if i < len(node_classes):
#                     current_classes = node_classes[i]
#                 else:
#                     current_classes = node_classes[-1]  # Assign remaining classes to the last node
#                 locs = [lab == joint_labels for lab in current_classes]
#                 combined_locs = torch.stack(locs).sum(0) > 0
#                 idx_keep = torch.nonzero(combined_locs).reshape(-1)
#                 train_subsets.append(torch.utils.data.Subset(full_train_set, idx_keep))
#         else:
#             raise NameError("Hetero data split N > number of classes not supported.")

#     # Create validation set
#     val_set = PointCloudDataset(all_points[:1000])  # Example validation set

#     # Create base models for each node
#     models = [PointNetAutoencoder() for _ in range(N)]

#     # Verify Model Dtypes
#     for idx, model in enumerate(models):
#         for name, param in model.named_parameters():
#             print(f"Model {idx}, Parameter {name}, dtype: {param.dtype}")

#     # Create DDLProblem instance
#     ddl_problem = DDLProblem(models=models, N=N, conf=conf, train_subsets=train_subsets, val_set=val_set)

#     # Define base loss function
#     if conf["loss"] == "Chamfer":
#         # Chamfer Distance is implemented directly in the loss computation
#         pass
#     else:
#         raise NameError("Unknown loss function.")

#     # Assign device
#     device = torch.device(conf["device"])
#     print(f"Device is set to {'GPU' if device.type == 'cuda' else 'CPU'}")

#     # Train using DiNNO
#     if conf["individual_training"]["train_solo"]:
#         print("Performing individual training...")
#         # Implement individual training logic here if needed
#     else:
#         try:
#             metrics = train_dinno(ddl_problem, None, val_set, graph, device, conf)
#         except Exception as e:
#             print(f"An error occurred during training: {e}")
#             metrics = None

#         if metrics is not None:
#             # Save metrics and models
#             torch.save(metrics, os.path.join(conf["output_metadir"], "dinno_metrics.pt"))
#             for idx, model in enumerate(ddl_problem.models):
#                 torch.save(model.state_dict(), os.path.join(conf["output_metadir"], f"dinno_trained_model_{idx}.pth"))
#             print("Training complete. Metrics and models saved.")

#             # Reconstruct and visualize the whole map with alignment
#             global_map = reconstruct_and_align_map(ddl_problem, device)
#             visualize_point_cloud(np.asarray(global_map.points), title="Reconstructed Global Map")

#             # Save the global map
#             o3d.io.write_point_cloud(os.path.join(conf["output_metadir"], "reconstructed_global_map.pcd"), global_map)
#             print("Reconstructed global map saved.")

import os
import sys
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import math
from datetime import datetime

# Check for CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is set to {'GPU' if DEVICE.type == 'cuda' else 'CPU'}")

# Attempt to import PyTorch3D if CUDA is available
USE_PYTORCH3D = False
if DEVICE.type == 'cuda':
    try:
        from pytorch3d.loss import chamfer_distance as pytorch3d_chamfer_distance
        USE_PYTORCH3D = True
        print("Using PyTorch3D's Chamfer Distance.")
    except ImportError:
        print("PyTorch3D not found. Falling back to pure PyTorch Chamfer Distance.")
        USE_PYTORCH3D = False
else:
    print("CUDA not available. Using pure PyTorch Chamfer Distance.")

# Define Chamfer Distance for CPU
def chamfer_distance_cpu(point_cloud1, point_cloud2):
    """
    Computes the Chamfer Distance between two point clouds on CPU.

    Args:
        point_cloud1: Tensor of shape (N, D), where N is the number of points and D is the dimensionality.
        point_cloud2: Tensor of shape (M, D).

    Returns:
        Chamfer Distance: Scalar tensor.
    """
    point_cloud1 = point_cloud1.unsqueeze(1)  # (N, 1, D)
    point_cloud2 = point_cloud2.unsqueeze(0)  # (1, M, D)

    # Compute pairwise squared distances
    distances = torch.sum((point_cloud1 - point_cloud2) ** 2, dim=2)  # (N, M)

    # For each point in point_cloud1, find the nearest point in point_cloud2
    min_dist1, _ = torch.min(distances, dim=1)  # (N,)

    # For each point in point_cloud2, find the nearest point in point_cloud1
    min_dist2, _ = torch.min(distances, dim=0)  # (M,)

    # Chamfer Distance is the sum of mean minimum distances
    chamfer_dist = torch.mean(min_dist1) + torch.mean(min_dist2)
    return chamfer_dist

# Define PointNet-Based Autoencoder
class PointNetAutoencoder(nn.Module):
    def __init__(self, num_points=1024):
        super(PointNetAutoencoder, self).__init__()
        self.num_points = num_points
        # Encoder
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        # Decoder
        self.fc3 = nn.Linear(256, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 3 * self.num_points)

    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))  # [batch_size, 64, num_points]
        x = F.relu(self.bn2(self.conv2(x)))  # [batch_size, 128, num_points]
        x = F.relu(self.bn3(self.conv3(x)))  # [batch_size, 1024, num_points]
        x = torch.max(x, 2, keepdim=True)[0]  # [batch_size, 1024, 1]
        x = x.view(-1, 1024)  # [batch_size, 1024]
        x = F.relu(self.bn4(self.fc1(x)))  # [batch_size, 512]
        x = F.relu(self.bn5(self.fc2(x)))  # [batch_size, 256]
        # Decoder
        x = F.relu(self.bn6(self.fc3(x)))  # [batch_size, 512]
        x = self.fc4(x)  # [batch_size, 3 * num_points]
        x = x.view(-1, 3, self.num_points)  # [batch_size, 3, num_points]
        return x

# Define Custom Dataset for Point Clouds
class PointCloudDataset(Dataset):
    def __init__(self, point_clouds, num_points=1024, augment=False):
        self.point_clouds = point_clouds
        self.augment = augment
        self.num_points = num_points
        # Normalize the point cloud
        self.point_clouds = (self.point_clouds - np.mean(self.point_clouds, axis=0)) / np.std(self.point_clouds, axis=0)

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        data = self.point_clouds[idx]
        if data.shape[0] < self.num_points:
            # If less points, pad with random points
            pad_size = self.num_points - data.shape[0]
            pad = np.random.randn(pad_size, 3) * 0.001
            data = np.vstack([data, pad])
        elif data.shape[0] > self.num_points:
            # If more points, randomly sample
            indices = np.random.choice(data.shape[0], self.num_points, replace=False)
            data = data[indices]
        if self.augment:
            # Apply random rotation around Z-axis
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,             0,              1]
            ])
            data = data @ rotation_matrix.T
        data = torch.tensor(data, dtype=torch.float32)
        return data, data  # Return a tuple (input, target)

# Placeholder for Graph Generation Utility
def generate_from_conf(graph_conf):
    """
    Generates a communication graph based on the configuration.

    Args:
        graph_conf: Dictionary containing graph configuration.

    Returns:
        N: Number of nodes.
        graph: NetworkX graph.
    """
    graph_type = graph_conf.get("type", "fully_connected")
    num_nodes = graph_conf.get("num_nodes", 10)
    p = graph_conf.get("p", 0.3)
    gen_attempts = graph_conf.get("gen_attempts", 100)

    if graph_type == "fully_connected":
        graph = nx.complete_graph(num_nodes)
    elif graph_type == "cycle":
        graph = nx.cycle_graph(num_nodes)
    elif graph_type == "ring":
        graph = nx.ring_graph(num_nodes)
    elif graph_type == "star":
        graph = nx.star_graph(num_nodes - 1)
    elif graph_type == "erdos_renyi":
        graph = nx.erdos_renyi_graph(num_nodes, p, seed=None, directed=False)
        attempts = 0
        while not nx.is_connected(graph) and attempts < gen_attempts:
            graph = nx.erdos_renyi_graph(num_nodes, p, seed=None, directed=False)
            attempts += 1
        if not nx.is_connected(graph):
            raise ValueError("Failed to generate a connected Erdos-Renyi graph.")
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    return num_nodes, graph

# Define DDLProblem Class
class DDLProblem:
    def __init__(self, models, N, conf, train_subsets, val_set):
        self.models = models
        self.N = N
        self.conf = conf
        # Calculate the total number of parameters in the first model
        self.n = torch.numel(torch.nn.utils.parameters_to_vector(self.models[0].parameters()))
        # Initialize communication graph
        self.graph = generate_from_conf(conf["graph"])[1]
        # Initialize data loaders for each node
        self.train_loaders = [
            DataLoader(train_subsets[i], batch_size=conf["train_batch_size"], shuffle=True)
            for i in range(N)
        ]
        self.val_loader = DataLoader(val_set, batch_size=conf["val_batch_size"], shuffle=False)
        # Assign device
        self.device = self.conf["device"]

    def local_batch_loss(self, i):
        """
        Compute the local batch loss for node i using the appropriate Chamfer Distance.
        """
        model = self.models[i].to(self.device)
        model.train()
        try:
            data, target = next(iter(self.train_loaders[i]))
        except StopIteration:
            # Restart the loader if the iterator is exhausted
            self.train_loaders[i] = iter(self.train_loaders[i])
            data, target = next(iter(self.train_loaders[i]))
        data, target = data.to(self.device), target.to(self.device)
        data = data.permute(0, 2, 1)  # [batch_size, 3, num_points]
        target = target.permute(0, 2, 1)  # [batch_size, 3, num_points]
        output = model(data)
        if USE_PYTORCH3D:
            loss_cd, _ = pytorch3d_chamfer_distance(output, target)
        else:
            # Reshape to (batch_size, num_points, 3)
            output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
            target = target.permute(0, 2, 1)  # [batch_size, num_points, 3]
            loss_cd = 0.0
            for j in range(output.size(0)):
                loss_cd += chamfer_distance_cpu(output[j], target[j])
            loss_cd = loss_cd / output.size(0)
        return loss_cd

    def evaluate_metrics(self, at_end=False, iteration=0):
        """
        Evaluate and return metrics such as validation loss using the appropriate Chamfer Distance.
        """
        metrics = {}
        for i, model in enumerate(self.models):
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    data = data.permute(0, 2, 1)  # [batch_size, 3, num_points]
                    target = target.permute(0, 2, 1)  # [batch_size, 3, num_points]
                    output = model(data)
                    if USE_PYTORCH3D:
                        loss_cd, _ = pytorch3d_chamfer_distance(output, target)
                    else:
                        # Reshape to (batch_size, num_points, 3)
                        output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
                        target = target.permute(0, 2, 1)  # [batch_size, num_points, 3]
                        loss_cd = 0.0
                        for j in range(output.size(0)):
                            loss_cd += chamfer_distance_cpu(output[j], target[j])
                        loss_cd = loss_cd / output.size(0)
                    total_loss += loss_cd.item()
            average_loss = total_loss / len(self.val_loader)
            metrics[f'validation_loss_node_{i}'] = average_loss
            print(f"Validation Loss for node {i}: {average_loss}")
        return metrics

    def update_graph(self):
        """
        Update the communication graph if needed.
        """
        # Implement any dynamic graph updates if required
        print("Updating communication graph...")
        # Example: No dynamic updates; keep the graph static
        pass

# Define DiNNO Optimizer Class
class DiNNO:
    def __init__(self, ddl_problem, device, conf):
        self.pr = ddl_problem
        self.conf = conf

        # Initialize dual variables
        self.duals = {
            i: torch.zeros((self.pr.n), device=device)
            for i in range(self.pr.N)
        }

        # Initialize penalty parameter rho
        self.rho = self.conf["rho_init"]
        self.rho_scaling = self.conf["rho_scaling"]

        # Learning rate scheduling
        if self.conf["lr_decay_type"] == "constant":
            self.primal_lr = self.conf["primal_lr_start"] * torch.ones(
                self.conf["outer_iterations"]
            )
        elif self.conf["lr_decay_type"] == "linear":
            self.primal_lr = torch.linspace(
                self.conf["primal_lr_start"],
                self.conf["primal_lr_finish"],
                self.conf["outer_iterations"],
            )
        elif self.conf["lr_decay_type"] == "log":
            self.primal_lr = torch.logspace(
                math.log(self.conf["primal_lr_start"], 10),
                math.log(self.conf["primal_lr_finish"], 10),
                self.conf["outer_iterations"],
            )
        else:
            raise ValueError("Unknown primal learning rate decay type.")

        self.pits = self.conf["primal_iterations"]

        # Initialize optimizers
        if self.conf["persistant_primal_opt"]:
            self.opts = {}
            for i in range(self.pr.N):
                if self.conf["primal_optimizer"] == "adam":
                    self.opts[i] = torch.optim.Adam(
                        self.pr.models[i].parameters(), self.primal_lr[0]
                    )
                elif self.conf["primal_optimizer"] == "sgd":
                    self.opts[i] = torch.optim.SGD(
                        self.pr.models[i].parameters(), self.primal_lr[0]
                    )
                elif self.conf["primal_optimizer"] == "adamw":
                    self.opts[i] = torch.optim.AdamW(
                        self.pr.models[i].parameters(), self.primal_lr[0]
                    )
                else:
                    raise ValueError("DiNNO primal optimizer is unknown.")

        # Initialize metrics storage
        self.metrics = []  # List to store metrics per epoch

    def primal_update(self, i, th_reg, k):
        if self.conf["persistant_primal_opt"]:
            opt = self.opts[i]
        else:
            if self.conf["primal_optimizer"] == "adam":
                opt = torch.optim.Adam(
                    self.pr.models[i].parameters(), self.primal_lr[k]
                )
            elif self.conf["primal_optimizer"] == "sgd":
                opt = torch.optim.SGD(
                    self.pr.models[i].parameters(), self.primal_lr[k]
                )
            elif self.conf["primal_optimizer"] == "adamw":
                opt = torch.optim.AdamW(
                    self.pr.models[i].parameters(), self.primal_lr[k]
                )
            else:
                raise ValueError("DiNNO primal optimizer is unknown.")

        for _ in range(self.pits):
            opt.zero_grad()

            # Model pass on the batch
            pred_loss = self.pr.local_batch_loss(i)

            # Get the primal variable WITH the autodiff graph attached.
            th = torch.nn.utils.parameters_to_vector(
                self.pr.models[i].parameters()
            )

            reg = torch.sum(
                torch.square(torch.cdist(th.reshape(1, -1), th_reg))
            )

            loss = pred_loss + torch.dot(th, self.duals[i]) + self.rho * reg
            loss.backward()
            opt.step()

        return

    def synchronize_weights(self):
        """
        Synchronize model weights with neighboring nodes by averaging.
        """
        for i in range(self.pr.N):
            neighbors = list(self.pr.graph.neighbors(i))
            if neighbors:
                # Collect weights from neighbors
                neighbor_weights = []
                for neighbor in neighbors:
                    neighbor_weights.append(torch.nn.utils.parameters_to_vector(
                        self.pr.models[neighbor].parameters()
                    ))
                # Average the weights
                avg_weights = torch.mean(torch.stack(neighbor_weights), dim=0)
                # Get current model weights
                current_weights = torch.nn.utils.parameters_to_vector(
                    self.pr.models[i].parameters()
                )
                # Update local model weights by averaging with neighbor's weights
                new_weights = (current_weights + avg_weights) / 2.0
                torch.nn.utils.vector_to_parameters(new_weights, self.pr.models[i].parameters())

    def train(self, profiler=None):
        eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
        oits = self.conf["outer_iterations"]
        for k in range(oits):
            if k % eval_every == 0 or k == oits - 1:
                # Evaluate metrics and append to self.metrics
                metrics = self.pr.evaluate_metrics(at_end=(k == oits - 1), iteration=k)
                self.metrics.append(metrics)  # Store metrics

            # Get the current primal variables
            ths = {
                i: torch.nn.utils.parameters_to_vector(
                    self.pr.models[i].parameters()
                )
                .clone()
                .detach()
                for i in range(self.pr.N)
            }

            # Update the penalty parameter
            self.rho *= self.rho_scaling

            # Update the communication graph
            self.pr.update_graph()

            # Per node updates
            for i in range(self.pr.N):
                neighs = list(self.pr.graph.neighbors(i))
                thj = torch.stack([ths[j] for j in neighs])

                self.duals[i] += self.rho * torch.sum(ths[i] - thj, dim=0)
                th_reg = (thj + ths[i]) / 2.0
                self.primal_update(i, th_reg, k)

            # Synchronize weights after each outer iteration
            self.synchronize_weights()

            if profiler is not None:
                profiler.step()

        return

# Define Reconstruction and Alignment Functions
def align_point_clouds(source_pcd, target_pcd, threshold=0.02):
    """
    Align source_pcd to target_pcd using ICP.

    Args:
        source_pcd: Open3D PointCloud object to be aligned.
        target_pcd: Open3D PointCloud object to align to.
        threshold: Distance threshold for ICP.

    Returns:
        Aligned source_pcd.
    """
    transformation = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    ).transformation
    source_pcd.transform(transformation)
    return source_pcd

def reconstruct_and_align_map(ddl_problem, device):
    """
    Reconstruct the entire map by aggregating and aligning local reconstructions from all nodes.

    Args:
        ddl_problem: Instance of DDLProblem containing models and data loaders.
        device: Torch device.

    Returns:
        global_map: Open3D PointCloud object representing the global map.
    """
    reconstructed_pcds = []
    for i in range(ddl_problem.N):
        model = ddl_problem.models[i].to(device)
        model.eval()
        all_reconstructions = []
        with torch.no_grad():
            for data, _ in ddl_problem.train_loaders[i]:
                data = data.to(device)
                data = data.permute(0, 2, 1)  # [batch_size, 3, num_points]
                output = model(data)
                if USE_PYTORCH3D:
                    # If using PyTorch3D, output is [batch_size, 3, num_points]
                    pass
                else:
                    # If using CPU, ensure output is [batch_size, 3, num_points]
                    pass
                output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
                all_reconstructions.append(output.cpu().numpy())
        reconstructed_points = np.concatenate(all_reconstructions, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(reconstructed_points)
        pcd = pcd.voxel_down_sample(voxel_size=0.05)  # Optional: Downsample for efficiency
        reconstructed_pcds.append(pcd)

    # Initialize global map with the first node's reconstruction
    global_map = reconstructed_pcds[0]

    for pcd in reconstructed_pcds[1:]:
        global_map = align_point_clouds(pcd, global_map)
        global_map += pcd
        global_map = global_map.voxel_down_sample(voxel_size=0.05)  # Optional: Downsample after merging

    return global_map

# Main Training Function
def train_dinno(ddl_problem, loss, val_set, graph, device, conf):
    # Define the DiNNO optimizer
    optimizer = DiNNO(ddl_problem, device, conf)

    # Start training
    optimizer.train()

    return optimizer.metrics

# Main Execution Block
if __name__ == "__main__":
    # Configuration
    conf = {
        "output_metadir": "./output/",
        "name": "3d_map_DiNNO",
        "epochs": 100,  # Corresponds to 'outer_iterations' in DiNNO
        "verbose": True,
        "graph": {
            "type": "cycle",  # Options: "fully_connected", "cycle", "ring", "star", "erdos_renyi"
            "num_nodes": 10,
            "p": 0.3,
            "gen_attempts": 100
        },
        "train_batch_size": 16,
        "val_batch_size": 16,
        "data_split_type": "random",  # Options: "random", "hetero"
        "data_dir": "/home/taherk/Downloads/2011_09_28_drive_0035_sync/2011_09_28/2011_09_28_drive_0035_sync/velodyne_points/data",  # Update this path
        "model": {
            "in_channels": 3,  # Updated for PointNet
            "out_channels": 3,
            "init_features": 3,
            "kernel_size": 1,
            "linear_width": 64
        },
        "loss": "Chamfer",  # Options: "Chamfer"
        "use_cuda": torch.cuda.is_available(),
        "individual_training": {
            "train_solo": False,
            "optimizer": "adam",
            "lr": 0.001,  # Reduced learning rate
            "verbose": True
        },
        # DiNNO Specific Hyperparameters
        "rho_init": 0.1,               # Initial rho value
        "rho_scaling": 1.1,            # Scaling factor for rho
        "lr_decay_type": "constant",   # 'constant', 'linear', or 'log'
        "primal_lr_start": 0.001,      # Starting learning rate for primal optimizer
        "primal_lr_finish": 0.0001,    # Final learning rate (used if lr_decay_type is 'linear' or 'log')
        "outer_iterations": 100,       # Number of outer iterations (set to 'epochs' value)
        "primal_iterations": 10,       # Number of primal updates per outer iteration
        "persistant_primal_opt": True,  # Use persistent primal optimizers
        "primal_optimizer": "adam",     # Type of primal optimizer: 'adam', 'sgd', 'adamw'
        "metrics_config": {             # Metrics configuration (if used)
            "evaluate_frequency": 1     # Evaluate metrics every iteration
        },
        "device": DEVICE.type,  # 'cuda' or 'cpu'
        "num_points": 1024  # Number of points per point cloud
    }

    # Create output directory
    if not os.path.exists(conf["output_metadir"]):
        os.makedirs(conf["output_metadir"], exist_ok=True)

    # Function to read KITTI Velodyne .bin file
    def read_kitti_bin(file_path):
        """
        Read a KITTI Velodyne .bin file and extract 3D point cloud (x, y, z).

        Args:
            file_path: Path to the .bin file.

        Returns:
            numpy array of shape (N, 3), where N is the number of points.
        """
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        return points[:, :3]  # Extract only x, y, z coordinates

    # Load point cloud data
    all_points = []
    for file_name in sorted(os.listdir(conf["data_dir"])):
        if file_name.endswith('.bin'):
            file_path = os.path.join(conf["data_dir"], file_name)
            points = read_kitti_bin(file_path)
            all_points.append(points)

    all_points = np.concatenate(all_points, axis=0)
    print(f"Total points loaded: {all_points.shape[0]}")

    # Visualize Original Point Cloud (optional)
    # visualize_point_cloud(all_points, title="Original Point Cloud")

    # Create communication graph
    try:
        N, graph = generate_from_conf(conf["graph"])
    except ValueError as e:
        print("Error:", str(e))
        available_graph_types = ["fully_connected", "cycle", "ring", "star", "erdos_renyi"]
        print("Available graph types:", available_graph_types)
        sys.exit(1)
    nx.write_gpickle(graph, os.path.join(conf["output_metadir"], "graph.gpickle"))
    print(f"Communication graph '{conf['graph']['type']}' with {N} nodes created.")

    # Create full training dataset
    full_train_set = PointCloudDataset(all_points, num_points=conf["num_points"], augment=True)

    # Split data into subsets for each node
    if conf["data_split_type"] == "random":
        num_samples_per = len(full_train_set) // N
        point_splits = [num_samples_per for _ in range(N - 1)]
        point_splits.append(len(full_train_set) - sum(point_splits))  # Ensure the split lengths sum to the total length
        train_subsets = torch.utils.data.random_split(full_train_set, point_splits)
        print("Data split randomly among nodes.")
    elif conf["data_split_type"] == "hetero":
        # Assuming the last column is class label; adjust if different
        # Note: KITTI Velodyne data does not have class labels; this is a placeholder
        raise NotImplementedError("Hetero data split not implemented.")
    else:
        raise ValueError(f"Unknown data_split_type: {conf['data_split_type']}")

    # Create validation set
    val_size = 1000  # Adjust as needed
    val_set = PointCloudDataset(all_points[:val_size], num_points=conf["num_points"], augment=False)
    print(f"Validation set size: {len(val_set)}")

    # Create base models for each node
    models = [PointNetAutoencoder(num_points=conf["num_points"]).to(DEVICE) for _ in range(N)]
    print(f"Created {N} PointNetAutoencoders.")

    # Verify Model Dtypes
    for idx, model in enumerate(models):
        for name, param in model.named_parameters():
            print(f"Model {idx}, Parameter {name}, dtype: {param.dtype}")

    # Create DDLProblem instance
    ddl_problem = DDLProblem(models=models, N=N, conf=conf, train_subsets=train_subsets, val_set=val_set)
    print("DDLProblem instance created.")

    # Define base loss function
    if conf["loss"] == "Chamfer":
        # Loss is handled within the DDLProblem class
        pass
    else:
        raise ValueError("Unknown loss function.")

    # Train using DiNNO
    if conf["individual_training"]["train_solo"]:
        print("Performing individual training...")
        # Implement individual training logic here if needed
        raise NotImplementedError("Individual training not implemented.")
    else:
        try:
            metrics = train_dinno(ddl_problem, None, val_set, graph, DEVICE, conf)
        except Exception as e:
            print(f"An error occurred during training: {e}")
            metrics = None

        if metrics is not None:
            # Save metrics and models
            torch.save(metrics, os.path.join(conf["output_metadir"], "dinno_metrics.pt"))
            for idx, model in enumerate(ddl_problem.models):
                torch.save(model.state_dict(), os.path.join(conf["output_metadir"], f"dinno_trained_model_{idx}.pth"))
            print("Training complete. Metrics and models saved.")

            # Reconstruct and visualize the whole map with alignment
            print("Reconstructing the global map...")
            global_map = reconstruct_and_align_map(ddl_problem, DEVICE)
            print("Global map reconstructed.")

            # Visualize the global map
            o3d.visualization.draw_geometries([global_map], window_name="Reconstructed Global Map")

            # Save the global map
            o3d.io.write_point_cloud(os.path.join(conf["output_metadir"], "reconstructed_global_map.pcd"), global_map)
            print("Reconstructed global map saved.")

    print("Script execution completed.")
