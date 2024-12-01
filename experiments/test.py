# import os
# import sys
# import glob
# import copy
# import math
# import random
# from datetime import datetime
# from shutil import copyfile

# import yaml
# import numpy as np
# import open3d as o3d
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import networkx as nx
# from torch.utils.data import Dataset, DataLoader

# # Optional: Uncomment if using PyTorch3D
# # from pytorch3d.loss import chamfer_distance as pytorch3d_chamfer_distance

# # Set random seeds for reproducibility
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# set_seed(42)

# # Configuration
# conf = {
#     "output_metadir": "output",
#     "name": "3d_map_DiNNO",
#     "epochs": 100,  # Corresponds to 'outer_iterations' in DiNNO
#     "verbose": True,
#     "graph": {
#         "type": "cycle",  # Options: "fully_connected", "cycle", "ring", "star", "erdos_renyi"
#         "num_nodes": 2,
#         "p": 0.3,
#         "gen_attempts": 100
#     },
#     "train_batch_size": 8,          
#     "val_batch_size": 8,            
#     "data_split_type": "spatial",   
#     "data_dir": "/home/taherk/nn_distributed_training/2011_09_28_drive_0035_sync/velodyne_points/data",
#     "model": {
#         "in_channels": 3,
#         "out_channels": 3,
#         "init_features": 3,
#         "kernel_size": 1,
#         "linear_width": 64
#     },
#     "loss": "Chamfer",  # Options: "Chamfer"
#     "use_cuda": torch.cuda.is_available(),
#     "individual_training": {
#         "train_solo": False,
#         "optimizer": "adam",
#         "lr": 0.0005,  # Adjusted learning rate
#         "verbose": True
#     },
#     # DiNNO Specific Hyperparameters
#     "rho_init": 0.05,               # Adjusted from 0.1
#     "rho_scaling": 1.05,            # Adjusted from 1.1
#     "lr_decay_type": "linear",      # Changed from "constant" to "linear"
#     "primal_lr_start": 0.0005,      # Adjusted from 0.001
#     "primal_lr_finish": 0.00005,    # Adjusted from 0.0001
#     "outer_iterations": 100,        # Number of outer iterations
#     "primal_iterations": 20,        # Increased from 10
#     "persistant_primal_opt": True,  # Use persistent primal optimizers
#     "primal_optimizer": "adam",     # Type of primal optimizer: 'adam', 'sgd', 'adamw'
#     "metrics_config": {             # Metrics configuration (if used)
#         "evaluate_frequency": 1     # Evaluate metrics every iteration
#     },
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "num_points": 1024               # Number of points per point cloud
# }

# DEVICE = torch.device(conf["device"])
# print(f"Using device: {DEVICE}")

# # Create output directory
# os.makedirs(conf["output_metadir"], exist_ok=True)

# # Function to read KITTI Velodyne .bin files
# def read_kitti_bin(file_path):
#     """
#     Read a KITTI Velodyne .bin file and extract 3D point cloud (x, y, z).

#     Args:
#         file_path: Path to the .bin file.

#     Returns:
#         numpy array of shape (N, 3), where N is the number of points.
#     """
#     points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
#     return points[:, :3]  # Extract only x, y, z coordinates

# # Function to visualize point clouds
# def visualize_point_clouds(original, reconstructed, title="Point Clouds"):
#     original_pcd = o3d.geometry.PointCloud()
#     original_pcd.points = o3d.utility.Vector3dVector(original)
#     original_pcd.paint_uniform_color([1, 0, 0])  # Red

#     reconstructed_pcd = o3d.geometry.PointCloud()
#     reconstructed_pcd.points = o3d.utility.Vector3dVector(reconstructed)
#     reconstructed_pcd.paint_uniform_color([0, 1, 0])  # Green

#     o3d.visualization.draw_geometries(
#         [original_pcd, reconstructed_pcd],
#         window_name=title
#     )

# # Custom Dataset for Point Clouds
# class PointCloudDataset(Dataset):
#     def __init__(self, point_clouds, num_points=1024, augment=False, mean=None, std=None):
#         self.point_clouds = point_clouds
#         self.augment = augment
#         self.num_points = num_points
        
#         if mean is None or std is None:
#             raise ValueError("Mean and Std must be provided for normalization.")
        
#         self.mean = mean
#         self.std = std
        
#         # Normalize the point cloud
#         self.point_clouds = (self.point_clouds - self.mean) / self.std

#     def __len__(self):
#         return len(self.point_clouds)

#     def __getitem__(self, idx):
#         data = self.point_clouds[idx]
#         if data.shape[0] < self.num_points:
#             pad_size = self.num_points - data.shape[0]
#             pad = np.random.randn(pad_size, 3) * 0.001
#             data = np.vstack([data, pad])
#         elif data.shape[0] > self.num_points:
#             indices = np.random.choice(data.shape[0], self.num_points, replace=False)
#             data = data[indices]
#         if self.augment:
#             # Controlled rotation around Z-axis
#             theta = np.random.uniform(0, 2 * np.pi)
#             rotation_matrix = np.array([
#                 [np.cos(theta), -np.sin(theta), 0],
#                 [np.sin(theta),  np.cos(theta), 0],
#                 [0,             0,              1]
#             ])
#             data = data @ rotation_matrix.T
#             # Optional: Controlled scaling
#             scale = np.random.uniform(0.95, 1.05)
#             data = data * scale
#         data = torch.tensor(data, dtype=torch.float32)
#         return data, data  # (input, target)

# # Enhanced PointNet Autoencoder
# class EnhancedPointNetAutoencoder(nn.Module):
#     def __init__(self, num_points=1024, num_groups=32):
#         super(EnhancedPointNetAutoencoder, self).__init__()
#         self.num_points = num_points
#         self.num_groups = num_groups

#         # Encoder
#         self.conv1 = nn.Conv1d(3, 64, 1)
#         self.gn1 = nn.GroupNorm(num_groups=self.num_groups, num_channels=64)
#         self.conv2 = nn.Conv1d(64, 128, 1)
#         self.gn2 = nn.GroupNorm(num_groups=self.num_groups, num_channels=128)
#         self.conv3 = nn.Conv1d(128, 256, 1)
#         self.gn3 = nn.GroupNorm(num_groups=self.num_groups, num_channels=256)
#         self.conv4 = nn.Conv1d(256, 512, 1)
#         self.gn4 = nn.GroupNorm(num_groups=self.num_groups, num_channels=512)
#         self.conv5 = nn.Conv1d(512, 1024, 1)
#         self.gn5 = nn.GroupNorm(num_groups=self.num_groups, num_channels=1024)
        
#         self.fc1 = nn.Linear(1024, 512)
#         self.gn6 = nn.GroupNorm(num_groups=self.num_groups, num_channels=512)
#         self.fc2 = nn.Linear(512, 256)
#         self.gn7 = nn.GroupNorm(num_groups=self.num_groups, num_channels=256)
        
#         # Decoder
#         self.fc3 = nn.Linear(256, 512)
#         self.gn8 = nn.GroupNorm(num_groups=self.num_groups, num_channels=512)
#         self.fc4 = nn.Linear(512, 1024)
#         self.gn9 = nn.GroupNorm(num_groups=self.num_groups, num_channels=1024)
#         self.fc5 = nn.Linear(1024, 3 * self.num_points)
        
#     def forward(self, x):
#         # Encoder
#         x1 = F.leaky_relu(self.gn1(self.conv1(x)), negative_slope=0.2)  # [B, 64, N]
#         x2 = F.leaky_relu(self.gn2(self.conv2(x1)), negative_slope=0.2)  # [B, 128, N]
#         x3 = F.leaky_relu(self.gn3(self.conv3(x2)), negative_slope=0.2)  # [B, 256, N]
#         x4 = F.leaky_relu(self.gn4(self.conv4(x3)), negative_slope=0.2)  # [B, 512, N]
#         x5 = F.leaky_relu(self.gn5(self.conv5(x4)), negative_slope=0.2)  # [B, 1024, N]
        
#         x = torch.max(x5, 2, keepdim=True)[0]  # [B, 1024, 1]
#         x = x.view(-1, 1024)  # [B, 1024]
#         x = F.leaky_relu(self.gn6(self.fc1(x)), negative_slope=0.2)  # [B, 512]
#         x = F.leaky_relu(self.gn7(self.fc2(x)), negative_slope=0.2)  # [B, 256]
        
#         # Decoder with skip connections
#         x = F.leaky_relu(self.gn8(self.fc3(x)), negative_slope=0.2)  # [B, 512]
#         x = F.leaky_relu(self.gn9(self.fc4(x)), negative_slope=0.2)  # [B, 1024]
#         x = self.fc5(x)  # [B, 3 * N]
#         x = x.view(-1, 3, self.num_points)  # [B, 3, N]
#         return x

# # Function to split point clouds spatially
# def spatial_split(point_clouds, num_regions, overlap_ratio=0.1):
#     """
#     Split the point cloud data into spatial regions for distributed training.

#     Args:
#         point_clouds: numpy array of shape (N, 3)
#         num_regions: Number of spatial regions (nodes)
#         overlap_ratio: Fraction of overlap between regions

#     Returns:
#         List of numpy arrays, each corresponding to a region
#     """
#     axis = 0  # Splitting along the X-axis
#     sorted_indices = np.argsort(point_clouds[:, axis])
#     sorted_points = point_clouds[sorted_indices]

#     total_range = sorted_points[:, axis].max() - sorted_points[:, axis].min()
#     region_size = total_range / num_regions
#     overlap_size = region_size * overlap_ratio

#     regions = []
#     min_coord = sorted_points[:, axis].min()
#     max_coord = sorted_points[:, axis].max()

#     for i in range(num_regions):
#         start = min_coord + i * region_size - (overlap_size if i > 0 else 0)
#         end = start + region_size + (overlap_size if i < num_regions - 1 else 0)

#         # Clamp the start and end to the min and max coordinates
#         start = max(start, min_coord)
#         end = min(end, max_coord)

#         region_mask = (sorted_points[:, axis] >= start) & (sorted_points[:, axis] < end)
#         region_points = sorted_points[region_mask]
#         regions.append(region_points)
#         print(f"Region {i}: {region_points.shape[0]} points, X between {start:.2f} and {end:.2f}")

#     return regions

# # Define Chamfer Distance for CPU (Batch-wise)
# def chamfer_distance_cpu_batch(point_cloud1, point_cloud2):
#     """
#     Computes the Chamfer Distance between two batches of point clouds on CPU.

#     Args:
#         point_cloud1: Tensor of shape (B, N, D)
#         point_cloud2: Tensor of shape (B, M, D)

#     Returns:
#         Chamfer Distance: Scalar tensor averaged over the batch.
#     """
#     B, N, D = point_cloud1.shape
#     M = point_cloud2.shape[1]

#     # Expand dimensions to compute pairwise distances
#     point_cloud1_exp = point_cloud1.unsqueeze(2)  # (B, N, 1, D)
#     point_cloud2_exp = point_cloud2.unsqueeze(1)  # (B, 1, M, D)

#     # Compute pairwise squared distances
#     distances = torch.sum((point_cloud1_exp - point_cloud2_exp) ** 2, dim=3)  # (B, N, M)

#     # For each point in point_cloud1, find the nearest point in point_cloud2
#     min_dist1, _ = torch.min(distances, dim=2)  # (B, N)

#     # For each point in point_cloud2, find the nearest point in point_cloud1
#     min_dist2, _ = torch.min(distances, dim=1)  # (B, M)

#     # Chamfer Distance is the sum of mean minimum distances
#     chamfer_dist = torch.mean(min_dist1) + torch.mean(min_dist2)
#     return chamfer_dist

# # Define the DDLProblem Class
# class DDLProblem:
#     def __init__(self, models, N, conf, train_subsets, val_set):
#         self.models = models  # List of models for each node
#         self.N = N            # Number of nodes
#         self.conf = conf
#         self.device = conf["device"]
#         self.num_points = conf["num_points"]

#         # Initialize Graph
#         self.graph = graph_generation(conf["graph"])

#         # Initialize Data Loaders
#         self.train_loaders = [DataLoader(dataset, batch_size=conf["train_batch_size"], shuffle=True, drop_last=True)
#                               for dataset in train_subsets]
#         self.val_loader = DataLoader(val_set, batch_size=conf["val_batch_size"], shuffle=False, drop_last=False)

#         # Initialize Iterators for training
#         self.train_iters = [iter(loader) for loader in self.train_loaders]

#     def local_batch_loss(self, i):
#         """
#         Compute the local batch loss for node i.

#         Args:
#             i: Node index

#         Returns:
#             loss_cd: Chamfer Distance loss
#         """
#         model = self.models[i].to(self.device)
#         model.train()
#         try:
#             data, target = next(self.train_iters[i])
#         except StopIteration:
#             self.train_iters[i] = iter(self.train_loaders[i])
#             data, target = next(self.train_iters[i])

#         data, target = data.to(self.device), target.to(self.device)
#         data = data.permute(0, 2, 1)  # [batch_size, 3, num_points]
#         target = target.permute(0, 2, 1)  # [batch_size, 3, num_points]
#         output = model(data)
#         output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]

#         loss_cd = chamfer_distance_cpu_batch(output, target.permute(0, 2, 1))
#         return loss_cd

#     def evaluate_metrics(self, at_end=False, iteration=0):
#         """
#         Evaluate and return metrics such as validation loss using the appropriate Chamfer Distance.
#         """
#         metrics = {}
#         for i, model in enumerate(self.models):
#             model.eval()
#             total_loss = 0.0
#             with torch.no_grad():
#                 for data, target in self.val_loader:
#                     data, target = data.to(self.device), target.to(self.device)
#                     data = data.permute(0, 2, 1)  # [batch_size, 3, num_points]
#                     target = target.permute(0, 2, 1)  # [batch_size, 3, num_points]
#                     output = model(data)
#                     output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
#                     loss_cd = chamfer_distance_cpu_batch(output, target.permute(0, 2, 1))
#                     total_loss += loss_cd.item()
#             average_loss = total_loss / len(self.val_loader)
#             metrics[f'validation_loss_node_{i}'] = average_loss
#             print(f"Validation Loss for node {i}: {average_loss:.6f}")
#         return metrics

#     def update_graph(self):
#         """
#         Update the communication graph if needed.
#         """
#         # Implement any dynamic graph updates if required
#         print("Updating communication graph...")
#         # Example: No dynamic updates; keep the graph static
#         pass

# # Define DiNNO Optimizer Class
# class DiNNO:
#     def __init__(self, ddl_problem, device, conf):
#         self.pr = ddl_problem
#         self.conf = conf

#         # Initialize dual variables
#         self.duals = {
#             i: torch.zeros((self.pr.num_points * 3), device=device)
#             for i in range(self.pr.N)
#         }

#         # Initialize penalty parameter rho
#         self.rho = self.conf["rho_init"]
#         self.rho_scaling = self.conf["rho_scaling"]

#         # Learning rate scheduling
#         if self.conf["lr_decay_type"] == "constant":
#             self.primal_lr = self.conf["primal_lr_start"] * torch.ones(
#                 self.conf["outer_iterations"]
#             )
#         elif self.conf["lr_decay_type"] == "linear":
#             self.primal_lr = torch.linspace(
#                 self.conf["primal_lr_start"],
#                 self.conf["primal_lr_finish"],
#                 self.conf["outer_iterations"],
#             )
#         elif self.conf["lr_decay_type"] == "log":
#             self.primal_lr = torch.logspace(
#                 math.log10(self.conf["primal_lr_start"]),
#                 math.log10(self.conf["primal_lr_finish"]),
#                 self.conf["outer_iterations"],
#             )
#         else:
#             raise ValueError("Unknown primal learning rate decay type.")

#         self.pits = self.conf["primal_iterations"]

#         # Initialize optimizers
#         if self.conf["persistant_primal_opt"]:
#             self.opts = {}
#             for i in range(self.pr.N):
#                 if self.conf["primal_optimizer"] == "adam":
#                     self.opts[i] = torch.optim.Adam(
#                         self.pr.models[i].parameters(), lr=self.primal_lr[0]
#                     )
#                 elif self.conf["primal_optimizer"] == "sgd":
#                     self.opts[i] = torch.optim.SGD(
#                         self.pr.models[i].parameters(), lr=self.primal_lr[0], momentum=0.9
#                     )
#                 elif self.conf["primal_optimizer"] == "adamw":
#                     self.opts[i] = torch.optim.AdamW(
#                         self.pr.models[i].parameters(), lr=self.primal_lr[0]
#                     )
#                 else:
#                     raise ValueError("DiNNO primal optimizer is unknown.")

#         # Initialize metrics storage
#         self.metrics = []  # List to store metrics per epoch

#         # Early Stopping parameters
#         self.best_loss = float('inf')
#         self.patience = 20  # Number of iterations to wait before stopping
#         self.counter = 0

#     def primal_update(self, i, th_reg, k):
#         if self.conf["persistant_primal_opt"]:
#             opt = self.opts[i]
#         else:
#             if self.conf["primal_optimizer"] == "adam":
#                 opt = torch.optim.Adam(
#                     self.pr.models[i].parameters(), lr=self.primal_lr[k]
#                 )
#             elif self.conf["primal_optimizer"] == "sgd":
#                 opt = torch.optim.SGD(
#                     self.pr.models[i].parameters(), lr=self.primal_lr[k], momentum=0.9
#                 )
#             elif self.conf["primal_optimizer"] == "adamw":
#                 opt = torch.optim.AdamW(
#                     self.pr.models[i].parameters(), lr=self.primal_lr[k]
#                 )
#             else:
#                 raise ValueError("DiNNO primal optimizer is unknown.")

#         for _ in range(self.pits):
#             opt.zero_grad()

#             # Model pass on the batch
#             pred_loss = self.pr.local_batch_loss(i)

#             # Get the primal variable WITH the autodiff graph attached.
#             th = torch.nn.utils.parameters_to_vector(
#                 self.pr.models[i].parameters()
#             )

#             reg = torch.sum(
#                 torch.square(torch.cdist(th.unsqueeze(0), th_reg.unsqueeze(0)))
#             )

#             loss = pred_loss + torch.dot(th, self.duals[i]) + self.rho * reg
#             loss.backward()

#             # Gradient clipping to prevent exploding gradients
#             torch.nn.utils.clip_grad_norm_(self.pr.models[i].parameters(), max_norm=1.0)

#             opt.step()

#     def synchronize_weights(self, epoch):
#         """
#         Synchronize model weights with neighboring nodes by blending local and neighbor weights.
#         """
#         alpha = 0.5  # Blending factor; can be adapted based on epoch or performance

#         for i in range(self.pr.N):
#             neighbors = list(self.pr.graph.neighbors(i))
#             if neighbors:
#                 # Collect weights from neighbors
#                 neighbor_weights = []
#                 for neighbor in neighbors:
#                     neighbor_weights.append(torch.nn.utils.parameters_to_vector(
#                         self.pr.models[neighbor].parameters()
#                     ))
#                 # Compute average of neighbor weights
#                 avg_weights = torch.mean(torch.stack(neighbor_weights), dim=0)
#                 # Get current model weights
#                 current_weights = torch.nn.utils.parameters_to_vector(
#                     self.pr.models[i].parameters()
#                 )
#                 # Blend local and neighbor weights
#                 new_weights = alpha * current_weights + (1 - alpha) * avg_weights
#                 # Update model weights
#                 torch.nn.utils.vector_to_parameters(new_weights, self.pr.models[i].parameters())
                
#                 if self.conf["verbose"]:
#                     print(f"Epoch {epoch}: Node {i} weights synchronized with neighbors using alpha={alpha}.")

#     def train(self, profiler=None):
#         eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
#         oits = self.conf["outer_iterations"]
#         for k in range(oits):
#             epoch_loss = 0.0  # Initialize epoch loss
#             num_batches = 0    # Initialize batch counter

#             for i in range(self.pr.N):
#                 loss_cd = self.pr.local_batch_loss(i)
#                 epoch_loss += loss_cd.item()
#                 num_batches += 1

#             avg_epoch_loss = epoch_loss / (self.pr.N * num_batches)
#             if self.conf["verbose"]:
#                 print(f"Iteration {k}, Average Training Loss: {avg_epoch_loss:.6f}")

#             if k % eval_every == 0 or k == oits - 1:
#                 metrics = self.pr.evaluate_metrics(at_end=(k == oits - 1), iteration=k)
#                 self.metrics.append(metrics)  # Store metrics

#                 # Calculate average validation loss across all nodes
#                 current_loss = np.mean(list(metrics.values()))

#                 # Early Stopping Check
#                 if current_loss < self.best_loss:
#                     self.best_loss = current_loss
#                     self.counter = 0
#                     print(f"Iteration {k}: Validation loss improved to {current_loss:.6f}")
#                 else:
#                     self.counter += 1
#                     print(f"Iteration {k}: No improvement in validation loss.")
#                     if self.counter >= self.patience:
#                         print("Early stopping triggered.")
#                         break

#             # Log rho value
#             if self.conf["verbose"]:
#                 print(f"Iteration {k}, Rho: {self.rho}")

#             # Get the current primal variables
#             ths = {
#                 i: torch.nn.utils.parameters_to_vector(
#                     self.pr.models[i].parameters()
#                 )
#                 .clone()
#                 .detach()
#                 for i in range(self.pr.N)
#             }

#             # Update the penalty parameter
#             self.rho *= self.rho_scaling

#             # Update the communication graph
#             self.pr.update_graph()

#             # Per node updates
#             for i in range(self.pr.N):
#                 neighs = list(self.pr.graph.neighbors(i))
#                 thj = torch.stack([ths[j] for j in neighs])
#                 thj_mean = torch.mean(thj, dim=0)

#                 self.duals[i] += self.rho * (ths[i] - thj_mean)
#                 th_reg = (thj_mean + ths[i]) / 2.0
#                 self.primal_update(i, th_reg, k)

#             # Synchronize weights after each outer iteration
#             self.synchronize_weights(k)

#             if profiler is not None:
#                 profiler.step()

#         return

# # Function to generate the communication graph
# def graph_generation(graph_conf):
#     """
#     Generate a communication graph based on the configuration.

#     Args:
#         graph_conf: Dictionary containing graph configuration.

#     Returns:
#         A NetworkX graph object.
#     """
#     graph_type = graph_conf["type"]
#     num_nodes = graph_conf["num_nodes"]
#     p = graph_conf.get("p", 0.3)
#     gen_attempts = graph_conf.get("gen_attempts", 100)

#     if graph_type == "fully_connected":
#         graph = nx.complete_graph(num_nodes)
#     elif graph_type == "cycle":
#         graph = nx.cycle_graph(num_nodes)
#     elif graph_type == "ring":
#         graph = nx.cycle_graph(num_nodes)
#     elif graph_type == "star":
#         graph = nx.star_graph(num_nodes - 1)
#     elif graph_type == "erdos_renyi":
#         graph = nx.erdos_renyi_graph(num_nodes, p, seed=42)
#         # Ensure the graph is connected
#         attempts = 0
#         while not nx.is_connected(graph) and attempts < gen_attempts:
#             graph = nx.erdos_renyi_graph(num_nodes, p, seed=42 + attempts)
#             attempts += 1
#         if not nx.is_connected(graph):
#             raise ValueError("Failed to generate a connected Erdős-Rényi graph.")
#     else:
#         raise ValueError("Unknown graph type.")

#     print(f"Generated {graph_type} graph with {num_nodes} nodes.")
#     return graph

# # Define the DDLProblem Class
# class DDLProblem:
#     def __init__(self, models, N, conf, train_subsets, val_set):
#         self.models = models  # List of models for each node
#         self.N = N            # Number of nodes
#         self.conf = conf
#         self.device = conf["device"]
#         self.num_points = conf["num_points"]

#         # Initialize Graph
#         self.graph = graph_generation(conf["graph"])

#         # Initialize Data Loaders
#         self.train_loaders = [DataLoader(dataset, batch_size=conf["train_batch_size"], shuffle=True, drop_last=True)
#                               for dataset in train_subsets]
#         self.val_loader = DataLoader(val_set, batch_size=conf["val_batch_size"], shuffle=False, drop_last=False)

#         # Initialize Iterators for training
#         self.train_iters = [iter(loader) for loader in self.train_loaders]

#     def local_batch_loss(self, i):
#         """
#         Compute the local batch loss for node i.

#         Args:
#             i: Node index

#         Returns:
#             loss_cd: Chamfer Distance loss
#         """
#         model = self.models[i].to(self.device)
#         model.train()
#         try:
#             data, target = next(self.train_iters[i])
#         except StopIteration:
#             self.train_iters[i] = iter(self.train_loaders[i])
#             data, target = next(self.train_iters[i])

#         data, target = data.to(self.device), target.to(self.device)
#         data = data.permute(0, 2, 1)  # [batch_size, 3, num_points]
#         target = target.permute(0, 2, 1)  # [batch_size, 3, num_points]
#         output = model(data)
#         output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]

#         loss_cd = chamfer_distance_cpu_batch(output, target.permute(0, 2, 1))
#         return loss_cd

#     def evaluate_metrics(self, at_end=False, iteration=0):
#         """
#         Evaluate and return metrics such as validation loss using the appropriate Chamfer Distance.
#         """
#         metrics = {}
#         for i, model in enumerate(self.models):
#             model.eval()
#             total_loss = 0.0
#             with torch.no_grad():
#                 for data, target in self.val_loader:
#                     data, target = data.to(self.device), target.to(self.device)
#                     data = data.permute(0, 2, 1)  # [batch_size, 3, num_points]
#                     target = target.permute(0, 2, 1)  # [batch_size, 3, num_points]
#                     output = model(data)
#                     output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
#                     loss_cd = chamfer_distance_cpu_batch(output, target.permute(0, 2, 1))
#                     total_loss += loss_cd.item()
#             average_loss = total_loss / len(self.val_loader)
#             metrics[f'validation_loss_node_{i}'] = average_loss
#             print(f"Validation Loss for node {i}: {average_loss:.6f}")
#         return metrics

#     def update_graph(self):
#         """
#         Update the communication graph if needed.
#         """
#         # Implement any dynamic graph updates if required
#         print("Updating communication graph...")
#         # Example: No dynamic updates; keep the graph static
#         pass

# # Define DiNNO Optimizer Class
# class DiNNO:
#     def __init__(self, ddl_problem, device, conf):
#         self.pr = ddl_problem
#         self.conf = conf

#         # Initialize dual variables
#         self.duals = {
#             i: torch.zeros((self.pr.num_points * 3), device=device)
#             for i in range(self.pr.N)
#         }

#         # Initialize penalty parameter rho
#         self.rho = self.conf["rho_init"]
#         self.rho_scaling = self.conf["rho_scaling"]

#         # Learning rate scheduling
#         if self.conf["lr_decay_type"] == "constant":
#             self.primal_lr = self.conf["primal_lr_start"] * torch.ones(
#                 self.conf["outer_iterations"]
#             )
#         elif self.conf["lr_decay_type"] == "linear":
#             self.primal_lr = torch.linspace(
#                 self.conf["primal_lr_start"],
#                 self.conf["primal_lr_finish"],
#                 self.conf["outer_iterations"],
#             )
#         elif self.conf["lr_decay_type"] == "log":
#             self.primal_lr = torch.logspace(
#                 math.log10(self.conf["primal_lr_start"]),
#                 math.log10(self.conf["primal_lr_finish"]),
#                 self.conf["outer_iterations"],
#             )
#         else:
#             raise ValueError("Unknown primal learning rate decay type.")

#         self.pits = self.conf["primal_iterations"]

#         # Initialize optimizers
#         if self.conf["persistant_primal_opt"]:
#             self.opts = {}
#             for i in range(self.pr.N):
#                 if self.conf["primal_optimizer"] == "adam":
#                     self.opts[i] = torch.optim.Adam(
#                         self.pr.models[i].parameters(), lr=self.primal_lr[0]
#                     )
#                 elif self.conf["primal_optimizer"] == "sgd":
#                     self.opts[i] = torch.optim.SGD(
#                         self.pr.models[i].parameters(), lr=self.primal_lr[0], momentum=0.9
#                     )
#                 elif self.conf["primal_optimizer"] == "adamw":
#                     self.opts[i] = torch.optim.AdamW(
#                         self.pr.models[i].parameters(), lr=self.primal_lr[0]
#                     )
#                 else:
#                     raise ValueError("DiNNO primal optimizer is unknown.")

#         # Initialize metrics storage
#         self.metrics = []  # List to store metrics per epoch

#         # Early Stopping parameters
#         self.best_loss = float('inf')
#         self.patience = 20  # Number of iterations to wait before stopping
#         self.counter = 0

#     def primal_update(self, i, th_reg, k):
#         if self.conf["persistant_primal_opt"]:
#             opt = self.opts[i]
#         else:
#             if self.conf["primal_optimizer"] == "adam":
#                 opt = torch.optim.Adam(
#                     self.pr.models[i].parameters(), lr=self.primal_lr[k]
#                 )
#             elif self.conf["primal_optimizer"] == "sgd":
#                 opt = torch.optim.SGD(
#                     self.pr.models[i].parameters(), lr=self.primal_lr[k], momentum=0.9
#                 )
#             elif self.conf["primal_optimizer"] == "adamw":
#                 opt = torch.optim.AdamW(
#                     self.pr.models[i].parameters(), lr=self.primal_lr[k]
#                 )
#             else:
#                 raise ValueError("DiNNO primal optimizer is unknown.")

#         for _ in range(self.pits):
#             opt.zero_grad()

#             # Model pass on the batch
#             pred_loss = self.pr.local_batch_loss(i)

#             # Get the primal variable WITH the autodiff graph attached.
#             th = torch.nn.utils.parameters_to_vector(
#                 self.pr.models[i].parameters()
#             )

#             reg = torch.sum(
#                 torch.square(torch.cdist(th.unsqueeze(0), th_reg.unsqueeze(0)))
#             )

#             loss = pred_loss + torch.dot(th, self.duals[i]) + self.rho * reg
#             loss.backward()

#             # Gradient clipping to prevent exploding gradients
#             torch.nn.utils.clip_grad_norm_(self.pr.models[i].parameters(), max_norm=1.0)

#             opt.step()

#     def synchronize_weights(self, epoch):
#         """
#         Synchronize model weights with neighboring nodes by blending local and neighbor weights.
#         """
#         alpha = 0.5  # Blending factor; can be adapted based on epoch or performance

#         for i in range(self.pr.N):
#             neighbors = list(self.pr.graph.neighbors(i))
#             if neighbors:
#                 # Collect weights from neighbors
#                 neighbor_weights = []
#                 for neighbor in neighbors:
#                     neighbor_weights.append(torch.nn.utils.parameters_to_vector(
#                         self.pr.models[neighbor].parameters()
#                     ))
#                 # Compute average of neighbor weights
#                 avg_weights = torch.mean(torch.stack(neighbor_weights), dim=0)
#                 # Get current model weights
#                 current_weights = torch.nn.utils.parameters_to_vector(
#                     self.pr.models[i].parameters()
#                 )
#                 # Blend local and neighbor weights
#                 new_weights = alpha * current_weights + (1 - alpha) * avg_weights
#                 # Update model weights
#                 torch.nn.utils.vector_to_parameters(new_weights, self.pr.models[i].parameters())
                
#                 if self.conf["verbose"]:
#                     print(f"Epoch {epoch}: Node {i} weights synchronized with neighbors using alpha={alpha}.")

#     def train(self, profiler=None):
#         eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
#         oits = self.conf["outer_iterations"]
#         for k in range(oits):
#             epoch_loss = 0.0  # Initialize epoch loss
#             num_batches = 0    # Initialize batch counter

#             for i in range(self.pr.N):
#                 loss_cd = self.pr.local_batch_loss(i)
#                 epoch_loss += loss_cd.item()
#                 num_batches += 1

#             avg_epoch_loss = epoch_loss / (self.pr.N * num_batches)
#             if self.conf["verbose"]:
#                 print(f"Iteration {k}, Average Training Loss: {avg_epoch_loss:.6f}")

#             if k % eval_every == 0 or k == oits - 1:
#                 metrics = self.pr.evaluate_metrics(at_end=(k == oits - 1), iteration=k)
#                 self.metrics.append(metrics)  # Store metrics

#                 # Calculate average validation loss across all nodes
#                 current_loss = np.mean(list(metrics.values()))

#                 # Early Stopping Check
#                 if current_loss < self.best_loss:
#                     self.best_loss = current_loss
#                     self.counter = 0
#                     print(f"Iteration {k}: Validation loss improved to {current_loss:.6f}")
#                 else:
#                     self.counter += 1
#                     print(f"Iteration {k}: No improvement in validation loss.")
#                     if self.counter >= self.patience:
#                         print("Early stopping triggered.")
#                         break

#             # Log rho value
#             if self.conf["verbose"]:
#                 print(f"Iteration {k}, Rho: {self.rho}")

#             # Get the current primal variables
#             ths = {
#                 i: torch.nn.utils.parameters_to_vector(
#                     self.pr.models[i].parameters()
#                 )
#                 .clone()
#                 .detach()
#                 for i in range(self.pr.N)
#             }

#             # Update the penalty parameter
#             self.rho *= self.rho_scaling

#             # Update the communication graph
#             self.pr.update_graph()

#             # Per node updates
#             for i in range(self.pr.N):
#                 neighs = list(self.pr.graph.neighbors(i))
#                 if neighs:
#                     thj = torch.stack([ths[j] for j in neighs])
#                     thj_mean = torch.mean(thj, dim=0)

#                     self.duals[i] += self.rho * (ths[i] - thj_mean)
#                     th_reg = (thj_mean + ths[i]) / 2.0
#                     self.primal_update(i, th_reg, k)

#             # Synchronize weights after each outer iteration
#             self.synchronize_weights(k)

#             if profiler is not None:
#                 profiler.step()

#         return

# # Define Reconstruction and Alignment Functions
# def align_point_clouds(source_pcd, target_pcd, threshold=0.02):
#     """
#     Align source_pcd to target_pcd using ICP.

#     Args:
#         source_pcd: Open3D PointCloud object to be aligned.
#         target_pcd: Open3D PointCloud object to align to.
#         threshold: Distance threshold for ICP.

#     Returns:
#         Aligned source_pcd.
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

#     Args:
#         ddl_problem: Instance of DDLProblem containing models and data loaders.
#         device: Torch device.

#     Returns:
#         global_map: Open3D PointCloud object representing the global map.
#     """
#     reconstructed_pcds = []
#     for i in range(ddl_problem.N):
#         model = ddl_problem.models[i].to(device)
#         model.eval()
#         all_reconstructions = []
#         with torch.no_grad():
#             for data, _ in ddl_problem.train_loaders[i]:
#                 data = data.to(device)
#                 data = data.permute(0, 2, 1)  # [batch_size, 3, num_points]
#                 output = model(data)
#                 output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
#                 all_reconstructions.append(output.cpu().numpy())

#         # Concatenate all reconstructions
#         reconstructed_points = np.concatenate(all_reconstructions, axis=0)

#         # Reshape to (n_points, 3) and convert to float64
#         reconstructed_points = reconstructed_points.reshape(-1, 3).astype(np.float64)

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

# # Function to generate the communication graph
# def graph_generation(graph_conf):
#     """
#     Generate a communication graph based on the configuration.

#     Args:
#         graph_conf: Dictionary containing graph configuration.

#     Returns:
#         A NetworkX graph object.
#     """
#     graph_type = graph_conf["type"]
#     num_nodes = graph_conf["num_nodes"]
#     p = graph_conf.get("p", 0.3)
#     gen_attempts = graph_conf.get("gen_attempts", 100)

#     if graph_type == "fully_connected":
#         graph = nx.complete_graph(num_nodes)
#     elif graph_type == "cycle":
#         graph = nx.cycle_graph(num_nodes)
#     elif graph_type == "ring":
#         graph = nx.cycle_graph(num_nodes)
#     elif graph_type == "star":
#         graph = nx.star_graph(num_nodes - 1)
#     elif graph_type == "erdos_renyi":
#         graph = nx.erdos_renyi_graph(num_nodes, p, seed=42)
#         # Ensure the graph is connected
#         attempts = 0
#         while not nx.is_connected(graph) and attempts < gen_attempts:
#             graph = nx.erdos_renyi_graph(num_nodes, p, seed=42 + attempts)
#             attempts += 1
#         if not nx.is_connected(graph):
#             raise ValueError("Failed to generate a connected Erdős-Rényi graph.")
#     else:
#         raise ValueError("Unknown graph type.")

#     print(f"Generated {graph_type} graph with {num_nodes} nodes.")
#     return graph

# # Main Training Function
# def train_dinno(ddl_problem, loss, val_set, graph, device, conf):
#     # Define the DiNNO optimizer
#     optimizer = DiNNO(ddl_problem, device, conf)

#     # Start training
#     optimizer.train()

#     return optimizer.metrics

# # Main Execution Block
# if __name__ == "__main__":
#     # Create output directory
#     if not os.path.exists(conf["output_metadir"]):
#         os.makedirs(conf["output_metadir"], exist_ok=True)

#     # Function to read KITTI bin files (defined earlier)
#     # Function to visualize point clouds (defined earlier)

#     # Load point cloud data
#     all_points = []
#     for file_name in sorted(os.listdir(conf["data_dir"])):
#         if file_name.endswith('.bin'):
#             file_path = os.path.join(conf["data_dir"], file_name)
#             points = read_kitti_bin(file_path)
#             all_points.append(points)

#     all_points = np.concatenate(all_points, axis=0)
#     print(f"Total points loaded: {all_points.shape[0]}")

#     # Compute global mean and std
#     global_mean = np.mean(all_points, axis=0)
#     global_std = np.std(all_points, axis=0)
#     print(f"Global Mean: {global_mean}")
#     print(f"Global Std: {global_std}")

#     # Visualize spatial regions (optional)
#     num_regions = conf["graph"]["num_nodes"]
#     overlap_ratio = 0.1
#     spatial_regions = spatial_split(all_points, num_regions, overlap_ratio=overlap_ratio)
#     # Optionally, visualize regions with distinct colors
#     # Example visualization for spatial split is omitted for brevity

#     # Create training subsets for each node with global normalization
#     train_subsets = []
#     for i in range(num_regions):
#         region_points = spatial_regions[i]
#         dataset = PointCloudDataset(
#             region_points, 
#             num_points=conf["num_points"], 
#             augment=True, 
#             mean=global_mean, 
#             std=global_std
#         )
#         train_subsets.append(dataset)
#         print(f"Node {i}: Training set size: {len(dataset)}")

#     print("Data split spatially among nodes with overlapping regions.")

#     # Create validation set with global normalization
#     val_region_points = spatial_regions[0]  # Adjust as needed
#     val_set = PointCloudDataset(
#         val_region_points, 
#         num_points=conf["num_points"], 
#         augment=False, 
#         mean=global_mean, 
#         std=global_std
#     )
#     print(f"Validation set size: {len(val_set)}")

#     # Initialize the enhanced model for each node
#     models = [EnhancedPointNetAutoencoder(num_points=conf["num_points"], num_groups=32).to(DEVICE) for _ in range(num_regions)]
#     print(f"Created {num_regions} Enhanced PointNetAutoencoders.")

#     # Verify Model Dtypes
#     for idx, model in enumerate(models):
#         for name, param in model.named_parameters():
#             print(f"Model {idx}, Parameter {name}, dtype: {param.dtype}")

#     # Create DDLProblem instance
#     ddl_problem = DDLProblem(models=models, N=num_regions, conf=conf, train_subsets=train_subsets, val_set=val_set)
#     print("DDLProblem instance created.")

#     # Define base loss function (Chamfer Distance is handled within DDLProblem)
#     if conf["loss"] == "Chamfer":
#         pass
#     else:
#         raise ValueError("Unknown loss function.")

#     # Train using DiNNO
#     if conf["individual_training"]["train_solo"]:
#         print("Performing individual training...")
#         # Implement individual training logic here if needed
#         raise NotImplementedError("Individual training not implemented.")
#     else:
#         try:
#             metrics = train_dinno(ddl_problem, None, val_set, None, DEVICE, conf)
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
#             print("Reconstructing the global map...")
#             global_map = reconstruct_and_align_map(ddl_problem, DEVICE)
#             print("Global map reconstructed.")

#             # Visualize the global map
#             o3d.visualization.draw_geometries([global_map], window_name="Reconstructed Global Map")

#             # Save the global map
#             o3d.io.write_point_cloud(os.path.join(conf["output_metadir"], "reconstructed_global_map.pcd"), global_map)
#             print("Reconstructed global map saved.")

#     print("Script execution completed.")

import os
import sys
import glob
import copy
import math
import random
from datetime import datetime
from shutil import copyfile

import yaml
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch.utils.data import Dataset, DataLoader

# Optional: Uncomment if using PyTorch3D
# from pytorch3d.loss import chamfer_distance as pytorch3d_chamfer_distance

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Configuration
conf = {
    "output_metadir": "output",
    "name": "3d_map_DiNNO",
    "epochs": 100,  # Corresponds to 'outer_iterations' in DiNNO
    "verbose": True,
    "graph": {
        "type": "cycle",  # Options: "fully_connected", "cycle", "ring", "star", "erdos_renyi"
        "num_nodes": 2,
        "p": 0.3,
        "gen_attempts": 100
    },
    "train_batch_size": 8,          
    "val_batch_size": 8,            
    "data_split_type": "spatial",   
    "data_dir": "/home/taherk/nn_distributed_training/2011_09_28_drive_0035_sync/velodyne_points/data",
    "model": {
        "in_channels": 3,
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
        "lr": 0.0005,  # Adjusted learning rate
        "verbose": True
    },
    # DiNNO Specific Hyperparameters
    "rho_init": 0.05,               # Adjusted from 0.1
    "rho_scaling": 1.05,            # Adjusted from 1.1
    "lr_decay_type": "linear",      # Changed from "constant" to "linear"
    "primal_lr_start": 0.0005,      # Adjusted from 0.001
    "primal_lr_finish": 0.00005,    # Adjusted from 0.0001
    "outer_iterations": 100,        # Number of outer iterations
    "primal_iterations": 20,        # Increased from 10
    "persistant_primal_opt": True,  # Use persistent primal optimizers
    "primal_optimizer": "adam",     # Type of primal optimizer: 'adam', 'sgd', 'adamw'
    "metrics_config": {             # Metrics configuration (if used)
        "evaluate_frequency": 1     # Evaluate metrics every iteration
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_points": 1024               # Number of points per point cloud
}

DEVICE = torch.device(conf["device"])
print(f"Using device: {DEVICE}")

# Create output directory
os.makedirs(conf["output_metadir"], exist_ok=True)

# Function to read KITTI Velodyne .bin files
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

# Function to visualize point clouds
def visualize_point_clouds(original, reconstructed, title="Point Clouds"):
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original)
    original_pcd.paint_uniform_color([1, 0, 0])  # Red

    reconstructed_pcd = o3d.geometry.PointCloud()
    reconstructed_pcd.points = o3d.utility.Vector3dVector(reconstructed)
    reconstructed_pcd.paint_uniform_color([0, 1, 0])  # Green

    o3d.visualization.draw_geometries(
        [original_pcd, reconstructed_pcd],
        window_name=title
    )

# Custom Dataset for Point Clouds
class PointCloudDataset(Dataset):
    def __init__(self, point_clouds, num_points=1024, augment=False, mean=None, std=None):
        self.point_clouds = point_clouds
        self.augment = augment
        self.num_points = num_points
        
        if mean is None or std is None:
            raise ValueError("Mean and Std must be provided for normalization.")
        
        self.mean = mean
        self.std = std
        
        # Normalize the point cloud
        self.point_clouds = (self.point_clouds - self.mean) / self.std

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        data = self.point_clouds[idx]
        if data.shape[0] < self.num_points:
            pad_size = self.num_points - data.shape[0]
            pad = np.random.randn(pad_size, 3) * 0.001
            data = np.vstack([data, pad])
        elif data.shape[0] > self.num_points:
            indices = np.random.choice(data.shape[0], self.num_points, replace=False)
            data = data[indices]
        if self.augment:
            # Controlled rotation around Z-axis
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,             0,              1]
            ])
            data = data @ rotation_matrix.T
            # Optional: Controlled scaling
            scale = np.random.uniform(0.95, 1.05)
            data = data * scale
        data = torch.tensor(data, dtype=torch.float32)
        return data, data  # (input, target)

# Enhanced PointNet Autoencoder
class EnhancedPointNetAutoencoder(nn.Module):
    def __init__(self, num_points=1024, num_groups=32):
        super(EnhancedPointNetAutoencoder, self).__init__()
        self.num_points = num_points
        self.num_groups = num_groups

        # Encoder
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.gn1 = nn.GroupNorm(num_groups=self.num_groups, num_channels=64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.gn2 = nn.GroupNorm(num_groups=self.num_groups, num_channels=128)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.gn3 = nn.GroupNorm(num_groups=self.num_groups, num_channels=256)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.gn4 = nn.GroupNorm(num_groups=self.num_groups, num_channels=512)
        self.conv5 = nn.Conv1d(512, 1024, 1)
        self.gn5 = nn.GroupNorm(num_groups=self.num_groups, num_channels=1024)
        
        self.fc1 = nn.Linear(1024, 512)
        self.gn6 = nn.GroupNorm(num_groups=self.num_groups, num_channels=512)
        self.fc2 = nn.Linear(512, 256)
        self.gn7 = nn.GroupNorm(num_groups=self.num_groups, num_channels=256)
        
        # Decoder
        self.fc3 = nn.Linear(256, 512)
        self.gn8 = nn.GroupNorm(num_groups=self.num_groups, num_channels=512)
        self.fc4 = nn.Linear(512, 1024)
        self.gn9 = nn.GroupNorm(num_groups=self.num_groups, num_channels=1024)
        self.fc5 = nn.Linear(1024, 3 * self.num_points)
        
    def forward(self, x):
        # Encoder
        x1 = F.leaky_relu(self.gn1(self.conv1(x)), negative_slope=0.2)  # [B, 64, N]
        x2 = F.leaky_relu(self.gn2(self.conv2(x1)), negative_slope=0.2)  # [B, 128, N]
        x3 = F.leaky_relu(self.gn3(self.conv3(x2)), negative_slope=0.2)  # [B, 256, N]
        x4 = F.leaky_relu(self.gn4(self.conv4(x3)), negative_slope=0.2)  # [B, 512, N]
        x5 = F.leaky_relu(self.gn5(self.conv5(x4)), negative_slope=0.2)  # [B, 1024, N]
        
        x = torch.max(x5, 2, keepdim=True)[0]  # [B, 1024, 1]
        x = x.view(-1, 1024)  # [B, 1024]
        x = F.leaky_relu(self.gn6(self.fc1(x)), negative_slope=0.2)  # [B, 512]
        x = F.leaky_relu(self.gn7(self.fc2(x)), negative_slope=0.2)  # [B, 256]
        
        # Decoder with skip connections
        x = F.leaky_relu(self.gn8(self.fc3(x)), negative_slope=0.2)  # [B, 512]
        x = F.leaky_relu(self.gn9(self.fc4(x)), negative_slope=0.2)  # [B, 1024]
        x = self.fc5(x)  # [B, 3 * N]
        x = x.view(-1, 3, self.num_points)  # [B, 3, N]
        return x

# Function to split point clouds spatially
def spatial_split(point_clouds, num_regions, overlap_ratio=0.1):
    """
    Split the point cloud data into spatial regions for distributed training.

    Args:
        point_clouds: numpy array of shape (N, 3)
        num_regions: Number of spatial regions (nodes)
        overlap_ratio: Fraction of overlap between regions

    Returns:
        List of numpy arrays, each corresponding to a region
    """
    axis = 0  # Splitting along the X-axis
    sorted_indices = np.argsort(point_clouds[:, axis])
    sorted_points = point_clouds[sorted_indices]

    total_range = sorted_points[:, axis].max() - sorted_points[:, axis].min()
    region_size = total_range / num_regions
    overlap_size = region_size * overlap_ratio

    regions = []
    min_coord = sorted_points[:, axis].min()
    max_coord = sorted_points[:, axis].max()

    for i in range(num_regions):
        start = min_coord + i * region_size - (overlap_size if i > 0 else 0)
        end = start + region_size + (overlap_size if i < num_regions - 1 else 0)

        # Clamp the start and end to the min and max coordinates
        start = max(start, min_coord)
        end = min(end, max_coord)

        region_mask = (sorted_points[:, axis] >= start) & (sorted_points[:, axis] < end)
        region_points = sorted_points[region_mask]
        regions.append(region_points)
        print(f"Region {i}: {region_points.shape[0]} points, X between {start:.2f} and {end:.2f}")

    return regions

# Define Chamfer Distance for CPU (Batch-wise)
def chamfer_distance_cpu_batch(point_cloud1, point_cloud2):
    """
    Computes the Chamfer Distance between two batches of point clouds on CPU.

    Args:
        point_cloud1: Tensor of shape (B, N, D)
        point_cloud2: Tensor of shape (B, M, D)

    Returns:
        Chamfer Distance: Scalar tensor averaged over the batch.
    """
    B, N, D = point_cloud1.shape
    M = point_cloud2.shape[1]

    # Expand dimensions to compute pairwise distances
    point_cloud1_exp = point_cloud1.unsqueeze(2)  # (B, N, 1, D)
    point_cloud2_exp = point_cloud2.unsqueeze(1)  # (B, 1, M, D)

    # Compute pairwise squared distances
    distances = torch.sum((point_cloud1_exp - point_cloud2_exp) ** 2, dim=3)  # (B, N, M)

    # For each point in point_cloud1, find the nearest point in point_cloud2
    min_dist1, _ = torch.min(distances, dim=2)  # (B, N)

    # For each point in point_cloud2, find the nearest point in point_cloud1
    min_dist2, _ = torch.min(distances, dim=1)  # (B, M)

    # Chamfer Distance is the sum of mean minimum distances
    chamfer_dist = torch.mean(min_dist1) + torch.mean(min_dist2)
    return chamfer_dist

# Define the DDLProblem Class
class DDLProblem:
    def __init__(self, models, N, conf, train_subsets, val_set):
        self.models = models  # List of models for each node
        self.N = N            # Number of nodes
        self.conf = conf
        self.device = conf["device"]
        self.num_points = conf["num_points"]

        # Initialize Graph
        self.graph = graph_generation(conf["graph"])

        # Initialize Data Loaders
        self.train_loaders = [DataLoader(dataset, batch_size=conf["train_batch_size"], shuffle=True, drop_last=True)
                              for dataset in train_subsets]
        self.val_loader = DataLoader(val_set, batch_size=conf["val_batch_size"], shuffle=False, drop_last=False)

        # Initialize Iterators for training
        self.train_iters = [iter(loader) for loader in self.train_loaders]

    def local_batch_loss(self, i):
        """
        Compute the local batch loss for node i.

        Args:
            i: Node index

        Returns:
            loss_cd: Chamfer Distance loss
        """
        model = self.models[i].to(self.device)
        model.train()
        try:
            data, target = next(self.train_iters[i])
        except StopIteration:
            self.train_iters[i] = iter(self.train_loaders[i])
            data, target = next(self.train_iters[i])

        data, target = data.to(self.device), target.to(self.device)
        data = data.permute(0, 2, 1)  # [batch_size, 3, num_points]
        target = target.permute(0, 2, 1)  # [batch_size, 3, num_points]
        output = model(data)
        output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]

        loss_cd = chamfer_distance_cpu_batch(output, target.permute(0, 2, 1))
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
                    output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
                    loss_cd = chamfer_distance_cpu_batch(output, target.permute(0, 2, 1))
                    total_loss += loss_cd.item()
            average_loss = total_loss / len(self.val_loader)
            metrics[f'validation_loss_node_{i}'] = average_loss
            print(f"Validation Loss for node {i}: {average_loss:.6f}")
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
            i: torch.zeros((self.pr.num_points * 3), device=device)
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
                math.log10(self.conf["primal_lr_start"]),
                math.log10(self.conf["primal_lr_finish"]),
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
                        self.pr.models[i].parameters(), lr=self.primal_lr[0]
                    )
                elif self.conf["primal_optimizer"] == "sgd":
                    self.opts[i] = torch.optim.SGD(
                        self.pr.models[i].parameters(), lr=self.primal_lr[0], momentum=0.9
                    )
                elif self.conf["primal_optimizer"] == "adamw":
                    self.opts[i] = torch.optim.AdamW(
                        self.pr.models[i].parameters(), lr=self.primal_lr[0]
                    )
                else:
                    raise ValueError("DiNNO primal optimizer is unknown.")

        # Initialize metrics storage
        self.metrics = []  # List to store metrics per epoch

        # Early Stopping parameters
        self.best_loss = float('inf')
        self.patience = 20  # Number of iterations to wait before stopping
        self.counter = 0

    def primal_update(self, i, th_reg, k):
        if self.conf["persistant_primal_opt"]:
            opt = self.opts[i]
        else:
            if self.conf["primal_optimizer"] == "adam":
                opt = torch.optim.Adam(
                    self.pr.models[i].parameters(), lr=self.primal_lr[k]
                )
            elif self.conf["primal_optimizer"] == "sgd":
                opt = torch.optim.SGD(
                    self.pr.models[i].parameters(), lr=self.primal_lr[k], momentum=0.9
                )
            elif self.conf["primal_optimizer"] == "adamw":
                opt = torch.optim.AdamW(
                    self.pr.models[i].parameters(), lr=self.primal_lr[k]
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
                torch.square(torch.cdist(th.unsqueeze(0), th_reg.unsqueeze(0)))
            )

            loss = pred_loss + torch.dot(th, self.duals[i]) + self.rho * reg
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.pr.models[i].parameters(), max_norm=1.0)

            opt.step()

    def synchronize_weights(self, epoch):
        """
        Synchronize model weights with neighboring nodes by blending local and neighbor weights.
        """
        alpha = 0.5  # Blending factor; can be adapted based on epoch or performance

        for i in range(self.pr.N):
            neighbors = list(self.pr.graph.neighbors(i))
            if neighbors:
                # Collect weights from neighbors
                neighbor_weights = []
                for neighbor in neighbors:
                    neighbor_weights.append(torch.nn.utils.parameters_to_vector(
                        self.pr.models[neighbor].parameters()
                    ))
                # Compute average of neighbor weights
                avg_weights = torch.mean(torch.stack(neighbor_weights), dim=0)
                # Get current model weights
                current_weights = torch.nn.utils.parameters_to_vector(
                    self.pr.models[i].parameters()
                )
                # Blend local and neighbor weights
                new_weights = alpha * current_weights + (1 - alpha) * avg_weights
                # Update model weights
                torch.nn.utils.vector_to_parameters(new_weights, self.pr.models[i].parameters())
                
                if self.conf["verbose"]:
                    print(f"Epoch {epoch}: Node {i} weights synchronized with neighbors using alpha={alpha}.")

    def train(self, profiler=None):
        eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
        oits = self.conf["outer_iterations"]
        for k in range(oits):
            epoch_loss = 0.0  # Initialize epoch loss
            num_batches = 0    # Initialize batch counter

            for i in range(self.pr.N):
                loss_cd = self.pr.local_batch_loss(i)
                epoch_loss += loss_cd.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / (self.pr.N * num_batches)
            if self.conf["verbose"]:
                print(f"Iteration {k}, Average Training Loss: {avg_epoch_loss:.6f}")

            if k % eval_every == 0 or k == oits - 1:
                metrics = self.pr.evaluate_metrics(at_end=(k == oits - 1), iteration=k)
                self.metrics.append(metrics)  # Store metrics

                # Calculate average validation loss across all nodes
                current_loss = np.mean(list(metrics.values()))

                # Early Stopping Check
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.counter = 0
                    print(f"Iteration {k}: Validation loss improved to {current_loss:.6f}")
                else:
                    self.counter += 1
                    print(f"Iteration {k}: No improvement in validation loss.")
                    if self.counter >= self.patience:
                        print("Early stopping triggered.")
                        break

            # Log rho value
            if self.conf["verbose"]:
                print(f"Iteration {k}, Rho: {self.rho}")

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
                if neighs:
                    thj = torch.stack([ths[j] for j in neighs])
                    thj_mean = torch.mean(thj, dim=0)

                    self.duals[i] += self.rho * (ths[i] - thj_mean)
                    th_reg = (thj_mean + ths[i]) / 2.0
                    self.primal_update(i, th_reg, k)

            # Synchronize weights after each outer iteration
            self.synchronize_weights(k)

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
                output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
                all_reconstructions.append(output.cpu().numpy())

        # Concatenate all reconstructions
        reconstructed_points = np.concatenate(all_reconstructions, axis=0)

        # Reshape to (n_points, 3) and convert to float64
        reconstructed_points = reconstructed_points.reshape(-1, 3).astype(np.float64)

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

# Function to generate the communication graph
def graph_generation(graph_conf):
    """
    Generate a communication graph based on the configuration.

    Args:
        graph_conf: Dictionary containing graph configuration.

    Returns:
        A NetworkX graph object.
    """
    graph_type = graph_conf["type"]
    num_nodes = graph_conf["num_nodes"]
    p = graph_conf.get("p", 0.3)
    gen_attempts = graph_conf.get("gen_attempts", 100)

    if graph_type == "fully_connected":
        graph = nx.complete_graph(num_nodes)
    elif graph_type == "cycle":
        graph = nx.cycle_graph(num_nodes)
    elif graph_type == "ring":
        graph = nx.cycle_graph(num_nodes)
    elif graph_type == "star":
        graph = nx.star_graph(num_nodes - 1)
    elif graph_type == "erdos_renyi":
        graph = nx.erdos_renyi_graph(num_nodes, p, seed=42)
        # Ensure the graph is connected
        attempts = 0
        while not nx.is_connected(graph) and attempts < gen_attempts:
            graph = nx.erdos_renyi_graph(num_nodes, p, seed=42 + attempts)
            attempts += 1
        if not nx.is_connected(graph):
            raise ValueError("Failed to generate a connected Erdős-Rényi graph.")
    else:
        raise ValueError("Unknown graph type.")

    print(f"Generated {graph_type} graph with {num_nodes} nodes.")
    return graph

# Define the DDLProblem Class
class DDLProblem:
    def __init__(self, models, N, conf, train_subsets, val_set):
        self.models = models  # List of models for each node
        self.N = N            # Number of nodes
        self.conf = conf
        self.device = conf["device"]
        self.num_points = conf["num_points"]

        # Initialize Graph
        self.graph = graph_generation(conf["graph"])

        # Initialize Data Loaders
        self.train_loaders = [DataLoader(dataset, batch_size=conf["train_batch_size"], shuffle=True, drop_last=True)
                              for dataset in train_subsets]
        self.val_loader = DataLoader(val_set, batch_size=conf["val_batch_size"], shuffle=False, drop_last=False)

        # Initialize Iterators for training
        self.train_iters = [iter(loader) for loader in self.train_loaders]

    def local_batch_loss(self, i):
        """
        Compute the local batch loss for node i.

        Args:
            i: Node index

        Returns:
            loss_cd: Chamfer Distance loss
        """
        model = self.models[i].to(self.device)
        model.train()
        try:
            data, target = next(self.train_iters[i])
        except StopIteration:
            self.train_iters[i] = iter(self.train_loaders[i])
            data, target = next(self.train_iters[i])

        data, target = data.to(self.device), target.to(self.device)
        data = data.permute(0, 2, 1)  # [batch_size, 3, num_points]
        target = target.permute(0, 2, 1)  # [batch_size, 3, num_points]
        output = model(data)
        output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]

        loss_cd = chamfer_distance_cpu_batch(output, target.permute(0, 2, 1))
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
                    output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
                    loss_cd = chamfer_distance_cpu_batch(output, target.permute(0, 2, 1))
                    total_loss += loss_cd.item()
            average_loss = total_loss / len(self.val_loader)
            metrics[f'validation_loss_node_{i}'] = average_loss
            print(f"Validation Loss for node {i}: {average_loss:.6f}")
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
            i: torch.zeros((self.pr.num_points * 3), device=device)
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
                math.log10(self.conf["primal_lr_start"]),
                math.log10(self.conf["primal_lr_finish"]),
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
                        self.pr.models[i].parameters(), lr=self.primal_lr[0]
                    )
                elif self.conf["primal_optimizer"] == "sgd":
                    self.opts[i] = torch.optim.SGD(
                        self.pr.models[i].parameters(), lr=self.primal_lr[0], momentum=0.9
                    )
                elif self.conf["primal_optimizer"] == "adamw":
                    self.opts[i] = torch.optim.AdamW(
                        self.pr.models[i].parameters(), lr=self.primal_lr[0]
                    )
                else:
                    raise ValueError("DiNNO primal optimizer is unknown.")

        # Initialize metrics storage
        self.metrics = []  # List to store metrics per epoch

        # Early Stopping parameters
        self.best_loss = float('inf')
        self.patience = 20  # Number of iterations to wait before stopping
        self.counter = 0

    def primal_update(self, i, th_reg, k):
        if self.conf["persistant_primal_opt"]:
            opt = self.opts[i]
        else:
            if self.conf["primal_optimizer"] == "adam":
                opt = torch.optim.Adam(
                    self.pr.models[i].parameters(), lr=self.primal_lr[k]
                )
            elif self.conf["primal_optimizer"] == "sgd":
                opt = torch.optim.SGD(
                    self.pr.models[i].parameters(), lr=self.primal_lr[k], momentum=0.9
                )
            elif self.conf["primal_optimizer"] == "adamw":
                opt = torch.optim.AdamW(
                    self.pr.models[i].parameters(), lr=self.primal_lr[k]
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
                torch.square(torch.cdist(th.unsqueeze(0), th_reg.unsqueeze(0)))
            )

            loss = pred_loss + torch.dot(th, self.duals[i]) + self.rho * reg
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.pr.models[i].parameters(), max_norm=1.0)

            opt.step()

    def synchronize_weights(self, epoch):
        """
        Synchronize model weights with neighboring nodes by blending local and neighbor weights.
        """
        alpha = 0.5  # Blending factor; can be adapted based on epoch or performance

        for i in range(self.pr.N):
            neighbors = list(self.pr.graph.neighbors(i))
            if neighbors:
                # Collect weights from neighbors
                neighbor_weights = []
                for neighbor in neighbors:
                    neighbor_weights.append(torch.nn.utils.parameters_to_vector(
                        self.pr.models[neighbor].parameters()
                    ))
                # Compute average of neighbor weights
                avg_weights = torch.mean(torch.stack(neighbor_weights), dim=0)
                # Get current model weights
                current_weights = torch.nn.utils.parameters_to_vector(
                    self.pr.models[i].parameters()
                )
                # Blend local and neighbor weights
                new_weights = alpha * current_weights + (1 - alpha) * avg_weights
                # Update model weights
                torch.nn.utils.vector_to_parameters(new_weights, self.pr.models[i].parameters())
                
                if self.conf["verbose"]:
                    print(f"Epoch {epoch}: Node {i} weights synchronized with neighbors using alpha={alpha}.")

    def train(self, profiler=None):
        eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
        oits = self.conf["outer_iterations"]
        for k in range(oits):
            epoch_loss = 0.0  # Initialize epoch loss
            num_batches = 0    # Initialize batch counter

            for i in range(self.pr.N):
                loss_cd = self.pr.local_batch_loss(i)
                epoch_loss += loss_cd.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / (self.pr.N * num_batches)
            if self.conf["verbose"]:
                print(f"Iteration {k}, Average Training Loss: {avg_epoch_loss:.6f}")

            if k % eval_every == 0 or k == oits - 1:
                metrics = self.pr.evaluate_metrics(at_end=(k == oits - 1), iteration=k)
                self.metrics.append(metrics)  # Store metrics

                # Calculate average validation loss across all nodes
                current_loss = np.mean(list(metrics.values()))

                # Early Stopping Check
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.counter = 0
                    print(f"Iteration {k}: Validation loss improved to {current_loss:.6f}")
                else:
                    self.counter += 1
                    print(f"Iteration {k}: No improvement in validation loss.")
                    if self.counter >= self.patience:
                        print("Early stopping triggered.")
                        break

            # Log rho value
            if self.conf["verbose"]:
                print(f"Iteration {k}, Rho: {self.rho}")

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
                if neighs:
                    thj = torch.stack([ths[j] for j in neighs])
                    thj_mean = torch.mean(thj, dim=0)

                    self.duals[i] += self.rho * (ths[i] - thj_mean)
                    th_reg = (thj_mean + ths[i]) / 2.0
                    self.primal_update(i, th_reg, k)

            # Synchronize weights after each outer iteration
            self.synchronize_weights(k)

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
                output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
                all_reconstructions.append(output.cpu().numpy())

        # Concatenate all reconstructions
        reconstructed_points = np.concatenate(all_reconstructions, axis=0)

        # Reshape to (n_points, 3) and convert to float64
        reconstructed_points = reconstructed_points.reshape(-1, 3).astype(np.float64)

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

# Function to generate the communication graph
def graph_generation(graph_conf):
    """
    Generate a communication graph based on the configuration.

    Args:
        graph_conf: Dictionary containing graph configuration.

    Returns:
        A NetworkX graph object.
    """
    graph_type = graph_conf["type"]
    num_nodes = graph_conf["num_nodes"]
    p = graph_conf.get("p", 0.3)
    gen_attempts = graph_conf.get("gen_attempts", 100)

    if graph_type == "fully_connected":
        graph = nx.complete_graph(num_nodes)
    elif graph_type == "cycle":
        graph = nx.cycle_graph(num_nodes)
    elif graph_type == "ring":
        graph = nx.cycle_graph(num_nodes)
    elif graph_type == "star":
        graph = nx.star_graph(num_nodes - 1)
    elif graph_type == "erdos_renyi":
        graph = nx.erdos_renyi_graph(num_nodes, p, seed=42)
        # Ensure the graph is connected
        attempts = 0
        while not nx.is_connected(graph) and attempts < gen_attempts:
            graph = nx.erdos_renyi_graph(num_nodes, p, seed=42 + attempts)
            attempts += 1
        if not nx.is_connected(graph):
            raise ValueError("Failed to generate a connected Erdős-Rényi graph.")
    else:
        raise ValueError("Unknown graph type.")

    print(f"Generated {graph_type} graph with {num_nodes} nodes.")
    return graph

# Define the DDLProblem Class
class DDLProblem:
    def __init__(self, models, N, conf, train_subsets, val_set):
        self.models = models  # List of models for each node
        self.N = N            # Number of nodes
        self.conf = conf
        self.device = conf["device"]
        self.num_points = conf["num_points"]

        # Initialize Graph
        self.graph = graph_generation(conf["graph"])

        # Initialize Data Loaders
        self.train_loaders = [DataLoader(dataset, batch_size=conf["train_batch_size"], shuffle=True, drop_last=True)
                              for dataset in train_subsets]
        self.val_loader = DataLoader(val_set, batch_size=conf["val_batch_size"], shuffle=False, drop_last=False)

        # Initialize Iterators for training
        self.train_iters = [iter(loader) for loader in self.train_loaders]

    def local_batch_loss(self, i):
        """
        Compute the local batch loss for node i.

        Args:
            i: Node index

        Returns:
            loss_cd: Chamfer Distance loss
        """
        model = self.models[i].to(self.device)
        model.train()
        try:
            data, target = next(self.train_iters[i])
        except StopIteration:
            self.train_iters[i] = iter(self.train_loaders[i])
            data, target = next(self.train_iters[i])

        data, target = data.to(self.device), target.to(self.device)
        data = data.permute(0, 2, 1)  # [batch_size, 3, num_points]
        target = target.permute(0, 2, 1)  # [batch_size, 3, num_points]
        output = model(data)
        output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]

        loss_cd = chamfer_distance_cpu_batch(output, target.permute(0, 2, 1))
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
                    output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
                    loss_cd = chamfer_distance_cpu_batch(output, target.permute(0, 2, 1))
                    total_loss += loss_cd.item()
            average_loss = total_loss / len(self.val_loader)
            metrics[f'validation_loss_node_{i}'] = average_loss
            print(f"Validation Loss for node {i}: {average_loss:.6f}")
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
            i: torch.zeros((self.pr.num_points * 3), device=device)
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
                math.log10(self.conf["primal_lr_start"]),
                math.log10(self.conf["primal_lr_finish"]),
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
                        self.pr.models[i].parameters(), lr=self.primal_lr[0]
                    )
                elif self.conf["primal_optimizer"] == "sgd":
                    self.opts[i] = torch.optim.SGD(
                        self.pr.models[i].parameters(), lr=self.primal_lr[0], momentum=0.9
                    )
                elif self.conf["primal_optimizer"] == "adamw":
                    self.opts[i] = torch.optim.AdamW(
                        self.pr.models[i].parameters(), lr=self.primal_lr[0]
                    )
                else:
                    raise ValueError("DiNNO primal optimizer is unknown.")

        # Initialize metrics storage
        self.metrics = []  # List to store metrics per epoch

        # Early Stopping parameters
        self.best_loss = float('inf')
        self.patience = 20  # Number of iterations to wait before stopping
        self.counter = 0

    def primal_update(self, i, th_reg, k):
        if self.conf["persistant_primal_opt"]:
            opt = self.opts[i]
        else:
            if self.conf["primal_optimizer"] == "adam":
                opt = torch.optim.Adam(
                    self.pr.models[i].parameters(), lr=self.primal_lr[k]
                )
            elif self.conf["primal_optimizer"] == "sgd":
                opt = torch.optim.SGD(
                    self.pr.models[i].parameters(), lr=self.primal_lr[k], momentum=0.9
                )
            elif self.conf["primal_optimizer"] == "adamw":
                opt = torch.optim.AdamW(
                    self.pr.models[i].parameters(), lr=self.primal_lr[k]
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
                torch.square(torch.cdist(th.unsqueeze(0), th_reg.unsqueeze(0)))
            )

            loss = pred_loss + torch.dot(th, self.duals[i]) + self.rho * reg
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.pr.models[i].parameters(), max_norm=1.0)

            opt.step()

    def synchronize_weights(self, epoch):
        """
        Synchronize model weights with neighboring nodes by blending local and neighbor weights.
        """
        alpha = 0.5  # Blending factor; can be adapted based on epoch or performance

        for i in range(self.pr.N):
            neighbors = list(self.pr.graph.neighbors(i))
            if neighbors:
                # Collect weights from neighbors
                neighbor_weights = []
                for neighbor in neighbors:
                    neighbor_weights.append(torch.nn.utils.parameters_to_vector(
                        self.pr.models[neighbor].parameters()
                    ))
                # Compute average of neighbor weights
                avg_weights = torch.mean(torch.stack(neighbor_weights), dim=0)
                # Get current model weights
                current_weights = torch.nn.utils.parameters_to_vector(
                    self.pr.models[i].parameters()
                )
                # Blend local and neighbor weights
                new_weights = alpha * current_weights + (1 - alpha) * avg_weights
                # Update model weights
                torch.nn.utils.vector_to_parameters(new_weights, self.pr.models[i].parameters())
                
                if self.conf["verbose"]:
                    print(f"Epoch {epoch}: Node {i} weights synchronized with neighbors using alpha={alpha}.")

    def train(self, profiler=None):
        eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
        oits = self.conf["outer_iterations"]
        for k in range(oits):
            epoch_loss = 0.0  # Initialize epoch loss
            num_batches = 0    # Initialize batch counter

            for i in range(self.pr.N):
                loss_cd = self.pr.local_batch_loss(i)
                epoch_loss += loss_cd.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / (self.pr.N * num_batches)
            if self.conf["verbose"]:
                print(f"Iteration {k}, Average Training Loss: {avg_epoch_loss:.6f}")

            if k % eval_every == 0 or k == oits - 1:
                metrics = self.pr.evaluate_metrics(at_end=(k == oits - 1), iteration=k)
                self.metrics.append(metrics)  # Store metrics

                # Calculate average validation loss across all nodes
                current_loss = np.mean(list(metrics.values()))

                # Early Stopping Check
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.counter = 0
                    print(f"Iteration {k}: Validation loss improved to {current_loss:.6f}")
                else:
                    self.counter += 1
                    print(f"Iteration {k}: No improvement in validation loss.")
                    if self.counter >= self.patience:
                        print("Early stopping triggered.")
                        break

            # Log rho value
            if self.conf["verbose"]:
                print(f"Iteration {k}, Rho: {self.rho}")

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
                if neighs:
                    thj = torch.stack([ths[j] for j in neighs])
                    thj_mean = torch.mean(thj, dim=0)

                    self.duals[i] += self.rho * (ths[i] - thj_mean)
                    th_reg = (thj_mean + ths[i]) / 2.0
                    self.primal_update(i, th_reg, k)

            # Synchronize weights after each outer iteration
            self.synchronize_weights(k)

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
                output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
                all_reconstructions.append(output.cpu().numpy())

        # Concatenate all reconstructions
        reconstructed_points = np.concatenate(all_reconstructions, axis=0)

        # Reshape to (n_points, 3) and convert to float64
        reconstructed_points = reconstructed_points.reshape(-1, 3).astype(np.float64)

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

# Function to generate the communication graph
def graph_generation(graph_conf):
    """
    Generate a communication graph based on the configuration.

    Args:
        graph_conf: Dictionary containing graph configuration.

    Returns:
        A NetworkX graph object.
    """
    graph_type = graph_conf["type"]
    num_nodes = graph_conf["num_nodes"]
    p = graph_conf.get("p", 0.3)
    gen_attempts = graph_conf.get("gen_attempts", 100)

    if graph_type == "fully_connected":
        graph = nx.complete_graph(num_nodes)
    elif graph_type == "cycle":
        graph = nx.cycle_graph(num_nodes)
    elif graph_type == "ring":
        graph = nx.cycle_graph(num_nodes)
    elif graph_type == "star":
        graph = nx.star_graph(num_nodes - 1)
    elif graph_type == "erdos_renyi":
        graph = nx.erdos_renyi_graph(num_nodes, p, seed=42)
        # Ensure the graph is connected
        attempts = 0
        while not nx.is_connected(graph) and attempts < gen_attempts:
            graph = nx.erdos_renyi_graph(num_nodes, p, seed=42 + attempts)
            attempts += 1
        if not nx.is_connected(graph):
            raise ValueError("Failed to generate a connected Erdős-Rényi graph.")
    else:
        raise ValueError("Unknown graph type.")

    print(f"Generated {graph_type} graph with {num_nodes} nodes.")
    return graph

# Main Training Function
def train_dinno(ddl_problem, loss, val_set, graph, device, conf):
    # Define the DiNNO optimizer
    optimizer = DiNNO(ddl_problem, device, conf)

    # Start training
    optimizer.train()

    return optimizer.metrics

# Main Execution Block
if __name__ == "__main__":
    # Create output directory
    if not os.path.exists(conf["output_metadir"]):
        os.makedirs(conf["output_metadir"], exist_ok=True)

    # Function to read KITTI bin files (defined earlier)
    # Function to visualize point clouds (defined earlier)

    # Load point cloud data
    all_points = []
    for file_name in sorted(os.listdir(conf["data_dir"])):
        if file_name.endswith('.bin'):
            file_path = os.path.join(conf["data_dir"], file_name)
            points = read_kitti_bin(file_path)
            all_points.append(points)

    all_points = np.concatenate(all_points, axis=0)
    print(f"Total points loaded: {all_points.shape[0]}")

    # Compute global mean and std
    global_mean = np.mean(all_points, axis=0)
    global_std = np.std(all_points, axis=0)
    print(f"Global Mean: {global_mean}")
    print(f"Global Std: {global_std}")

    # Define number of regions (for potential future distributed training)
    num_regions = conf["graph"]["num_nodes"]
    overlap_ratio = 0.1

    # Spatial splitting (for future distributed training)
    spatial_regions = spatial_split(all_points, num_regions, overlap_ratio=overlap_ratio)
    # Optionally, visualize regions with distinct colors
    # Example visualization for spatial split is omitted for brevity

    # Create training subsets for each node with global normalization
    train_subsets = []
    for i in range(num_regions):
        region_points = spatial_regions[i]
        dataset = PointCloudDataset(
            region_points, 
            num_points=conf["num_points"], 
            augment=False,  # Data augmentation disabled
            mean=global_mean, 
            std=global_std
        )
        train_subsets.append(dataset)
        print(f"Node {i}: Training set size: {len(dataset)}")

    print("Data split spatially among nodes with overlapping regions and data augmentation disabled.")

    # Create validation set with global normalization
    val_region_points = spatial_regions[0]  # Adjust as needed
    val_set = PointCloudDataset(
        val_region_points, 
        num_points=conf["num_points"], 
        augment=False, 
        mean=global_mean, 
        std=global_std
    )
    print(f"Validation set size: {len(val_set)}")

    # Initialize the enhanced model for each node
    models = [EnhancedPointNetAutoencoder(num_points=conf["num_points"], num_groups=32).to(DEVICE) for _ in range(num_regions)]
    print(f"Created {num_regions} Enhanced PointNetAutoencoders.")

    # Verify Model Dtypes
    for idx, model in enumerate(models):
        for name, param in model.named_parameters():
            print(f"Model {idx}, Parameter {name}, dtype: {param.dtype}")

    # Create DDLProblem instance
    ddl_problem = DDLProblem(models=models, N=num_regions, conf=conf, train_subsets=train_subsets, val_set=val_set)
    print("DDLProblem instance created.")

    # Define base loss function (Chamfer Distance is handled within DDLProblem)
    if conf["loss"] == "Chamfer":
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
            metrics = train_dinno(ddl_problem, None, val_set, None, DEVICE, conf)
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
