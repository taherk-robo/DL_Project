# # # import os
# # # import sys
# # # import numpy as np
# # # import open3d as o3d
# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F
# # # from torch.utils.data import Dataset, DataLoader
# # # import networkx as nx
# # # import math
# # # from datetime import datetime
# # # import matplotlib.pyplot as plt  # For visualization

# # # # Check for CUDA availability
# # # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # print(f"Device is set to {'GPU' if DEVICE.type == 'cuda' else 'CPU'}")

# # # # Attempt to import PyTorch3D if CUDA is available
# # # USE_PYTORCH3D = False
# # # if DEVICE.type == 'cuda':
# # #     try:
# # #         from pytorch3d.loss import chamfer_distance as pytorch3d_chamfer_distance
# # #         USE_PYTORCH3D = True
# # #         print("Using PyTorch3D's Chamfer Distance.")
# # #     except ImportError:
# # #         print("PyTorch3D not found. Falling back to pure PyTorch Chamfer Distance.")
# # #         USE_PYTORCH3D = False
# # # else:
# # #     print("CUDA not available. Using pure PyTorch Chamfer Distance.")

# # # # Define Chamfer Distance for CPU
# # # def chamfer_distance_cpu(point_cloud1, point_cloud2):
# # #     """
# # #     Computes the Chamfer Distance between two point clouds on CPU.

# # #     Args:
# # #         point_cloud1: Tensor of shape (N, D), where N is the number of points and D is the dimensionality.
# # #         point_cloud2: Tensor of shape (M, D).

# # #     Returns:
# # #         Chamfer Distance: Scalar tensor.
# # #     """
# # #     point_cloud1 = point_cloud1.unsqueeze(1)  # (N, 1, D)
# # #     point_cloud2 = point_cloud2.unsqueeze(0)  # (1, M, D)

# # #     # Compute pairwise squared distances
# # #     distances = torch.sum((point_cloud1 - point_cloud2) ** 2, dim=2)  # (N, M)

# # #     # For each point in point_cloud1, find the nearest point in point_cloud2
# # #     min_dist1, _ = torch.min(distances, dim=1)  # (N,)

# # #     # For each point in point_cloud2, find the nearest point in point_cloud1
# # #     min_dist2, _ = torch.min(distances, dim=0)  # (M,)

# # #     # Chamfer Distance is the sum of mean minimum distances
# # #     chamfer_dist = torch.mean(min_dist1) + torch.mean(min_dist2)
# # #     return chamfer_dist

# # # # Define PointNet-Based Autoencoder with GroupNorm
# # # class PointNetAutoencoder(nn.Module):
# # #     def __init__(self, num_points=1024, num_groups=32):
# # #         super(PointNetAutoencoder, self).__init__()
# # #         self.num_points = num_points
# # #         self.num_groups = num_groups  # Number of groups for GroupNorm

# # #         # Encoder
# # #         self.conv1 = nn.Conv1d(3, 64, 1)
# # #         self.gn1 = nn.GroupNorm(num_groups=self.num_groups, num_channels=64)
# # #         self.conv2 = nn.Conv1d(64, 128, 1)
# # #         self.gn2 = nn.GroupNorm(num_groups=self.num_groups, num_channels=128)
# # #         self.conv3 = nn.Conv1d(128, 1024, 1)
# # #         self.gn3 = nn.GroupNorm(num_groups=self.num_groups, num_channels=1024)
# # #         self.fc1 = nn.Linear(1024, 512)
# # #         self.gn4 = nn.GroupNorm(num_groups=self.num_groups, num_channels=512)
# # #         self.fc2 = nn.Linear(512, 256)
# # #         self.gn5 = nn.GroupNorm(num_groups=self.num_groups, num_channels=256)
        
# # #         # Decoder
# # #         self.fc3 = nn.Linear(256, 512)
# # #         self.gn6 = nn.GroupNorm(num_groups=self.num_groups, num_channels=512)
# # #         self.fc4 = nn.Linear(512, 3 * self.num_points)

# # #     def forward(self, x):
# # #         # Encoder
# # #         x = F.relu(self.gn1(self.conv1(x)))  # [batch_size, 64, num_points]
# # #         x = F.relu(self.gn2(self.conv2(x)))  # [batch_size, 128, num_points]
# # #         x = F.relu(self.gn3(self.conv3(x)))  # [batch_size, 1024, num_points]
# # #         x = torch.max(x, 2, keepdim=True)[0]  # [batch_size, 1024, 1]
# # #         x = x.view(-1, 1024)  # [batch_size, 1024]
# # #         x = F.relu(self.gn4(self.fc1(x)))  # [batch_size, 512]
# # #         x = F.relu(self.gn5(self.fc2(x)))  # [batch_size, 256]
        
# # #         # Decoder
# # #         x = F.relu(self.gn6(self.fc3(x)))  # [batch_size, 512]
# # #         x = self.fc4(x)  # [batch_size, 3 * num_points]
# # #         x = x.view(-1, 3, self.num_points)  # [batch_size, 3, num_points]
# # #         return x

# # # # Define Custom Dataset for Point Clouds
# # # class PointCloudDataset(Dataset):
# # #     def __init__(self, point_cloud_list, num_points=1024, augment=False):
# # #         """
# # #         Args:
# # #             point_cloud_list: List of numpy arrays, each of shape (N_i, 3).
# # #             num_points: Number of points to sample from each point cloud.
# # #             augment: Whether to apply data augmentation.
# # #         """
# # #         self.point_cloud_list = point_cloud_list
# # #         self.augment = augment
# # #         self.num_points = num_points

# # #         # Compute overall mean and std for normalization
# # #         all_points = np.concatenate(self.point_cloud_list, axis=0)
# # #         self.mean = np.mean(all_points, axis=0)
# # #         self.std = np.std(all_points, axis=0)

# # #         # Normalize each point cloud
# # #         self.point_cloud_list = [
# # #             (pc - self.mean) / self.std for pc in self.point_cloud_list
# # #         ]

# # #     def __len__(self):
# # #         return len(self.point_cloud_list)

# # #     def __getitem__(self, idx):
# # #         point_cloud = self.point_cloud_list[idx]
# # #         num_points_in_cloud = point_cloud.shape[0]

# # #         if num_points_in_cloud >= self.num_points:
# # #             # Randomly sample num_points
# # #             indices = np.random.choice(num_points_in_cloud, self.num_points, replace=False)
# # #             data = point_cloud[indices]
# # #         else:
# # #             # If fewer points, pad with random points
# # #             pad_size = self.num_points - num_points_in_cloud
# # #             pad = np.random.normal(0, 0.001, size=(pad_size, 3))
# # #             data = np.concatenate([point_cloud, pad], axis=0)

# # #         if self.augment:
# # #             # Apply random rotation around Z-axis
# # #             theta = np.random.uniform(0, 2 * np.pi)
# # #             rotation_matrix = np.array([
# # #                 [np.cos(theta), -np.sin(theta), 0],
# # #                 [np.sin(theta),  np.cos(theta), 0],
# # #                 [0,              0,             1]
# # #             ])
# # #             data = data @ rotation_matrix.T

# # #         data = torch.tensor(data, dtype=torch.float32)
# # #         data = data.T  # Transpose to shape (3, num_points)
# # #         return data, data  # Return (input, target)

# # # # Function to perform spatial splitting (if needed)
# # # def spatial_split(point_clouds, num_regions, overlap_ratio=0.1):
# # #     """
# # #     Splits point clouds into spatial regions with overlapping areas.

# # #     Args:
# # #         point_clouds (list of numpy.ndarray): List of point clouds.
# # #         num_regions (int): Number of regions (robots).
# # #         overlap_ratio (float): Fraction of overlap between adjacent regions.
        
# # #     Returns:
# # #         List of lists: Each sublist contains point clouds assigned to that region.
# # #     """
# # #     # For this example, we'll split the point clouds evenly among regions
# # #     total_point_clouds = len(point_clouds)
# # #     point_clouds_per_region = total_point_clouds // num_regions
# # #     regions = []
# # #     for i in range(num_regions):
# # #         start_idx = i * point_clouds_per_region
# # #         end_idx = (i + 1) * point_clouds_per_region if i < num_regions - 1 else total_point_clouds
# # #         region_point_clouds = point_clouds[start_idx:end_idx]
# # #         regions.append(region_point_clouds)
# # #         print(f"Region {i}: {len(region_point_clouds)} point clouds.")
# # #     return regions

# # # # Placeholder for Graph Generation Utility
# # # def generate_from_conf(graph_conf):
# # #     """
# # #     Generates a communication graph based on the configuration.

# # #     Args:
# # #         graph_conf: Dictionary containing graph configuration.

# # #     Returns:
# # #         N: Number of nodes.
# # #         graph: NetworkX graph.
# # #     """
# # #     graph_type = graph_conf.get("type", "fully_connected")
# # #     num_nodes = graph_conf.get("num_nodes", 10)
# # #     p = graph_conf.get("p", 0.3)
# # #     gen_attempts = graph_conf.get("gen_attempts", 100)

# # #     if graph_type == "fully_connected":
# # #         graph = nx.complete_graph(num_nodes)
# # #     elif graph_type == "cycle":
# # #         graph = nx.cycle_graph(num_nodes)
# # #     elif graph_type == "ring":
# # #         graph = nx.cycle_graph(num_nodes)
# # #     elif graph_type == "star":
# # #         graph = nx.star_graph(num_nodes - 1)
# # #     elif graph_type == "erdos_renyi":
# # #         graph = nx.erdos_renyi_graph(num_nodes, p, seed=None, directed=False)
# # #         attempts = 0
# # #         while not nx.is_connected(graph) and attempts < gen_attempts:
# # #             graph = nx.erdos_renyi_graph(num_nodes, p, seed=None, directed=False)
# # #             attempts += 1
# # #         if not nx.is_connected(graph):
# # #             raise ValueError("Failed to generate a connected Erdos-Renyi graph.")
# # #     else:
# # #         raise ValueError(f"Unknown graph type: {graph_type}")

# # #     return num_nodes, graph

# # # # Define DDLProblem Class
# # # class DDLProblem:
# # #     def __init__(self, models, N, conf, train_subsets, val_set):
# # #         self.models = models
# # #         self.N = N
# # #         self.conf = conf
# # #         # Calculate the total number of parameters in the first model
# # #         self.n = torch.numel(torch.nn.utils.parameters_to_vector(self.models[0].parameters()))
# # #         # Initialize communication graph
# # #         self.graph = generate_from_conf(conf["graph"])[1]
# # #         # Initialize data loaders for each node with drop_last=False to retain all data
# # #         self.train_loaders = [
# # #             DataLoader(
# # #                 train_subsets[i],
# # #                 batch_size=conf["train_batch_size"],
# # #                 shuffle=True,
# # #                 drop_last=False  # Ensures all data is processed
# # #             )
# # #             for i in range(N)
# # #         ]
# # #         self.val_loader = DataLoader(
# # #             val_set,
# # #             batch_size=conf["val_batch_size"],
# # #             shuffle=False,
# # #             drop_last=False  # Ensures all validation data is processed
# # #         )
# # #         # Assign device
# # #         self.device = torch.device(conf["device"])
# # #         # Initialize iterators for each train loader
# # #         self.train_iters = [iter(loader) for loader in self.train_loaders]

# # #     def local_batch_loss(self, i):
# # #         """
# # #         Compute the local batch loss for node i using the appropriate Chamfer Distance.
# # #         """
# # #         model = self.models[i].to(self.device)
# # #         model.train()
# # #         try:
# # #             data, target = next(self.train_iters[i])
# # #         except StopIteration:
# # #             # Restart the loader if the iterator is exhausted
# # #             self.train_iters[i] = iter(self.train_loaders[i])
# # #             data, target = next(self.train_iters[i])
# # #         data, target = data.to(self.device), target.to(self.device)
# # #         output = model(data)
# # #         if USE_PYTORCH3D:
# # #             loss_cd, _ = pytorch3d_chamfer_distance(output.permute(0, 2, 1), target.permute(0, 2, 1))
# # #         else:
# # #             # Reshape to (batch_size, num_points, 3)
# # #             output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
# # #             target = target.permute(0, 2, 1)  # [batch_size, num_points, 3]
# # #             loss_cd = 0.0
# # #             for j in range(output.size(0)):
# # #                 loss_cd += chamfer_distance_cpu(output[j], target[j])
# # #             loss_cd = loss_cd / output.size(0)
# # #         return loss_cd

# # #     def evaluate_metrics(self, at_end=False, iteration=0):
# # #         """
# # #         Evaluate and return metrics such as validation loss using the appropriate Chamfer Distance.
# # #         """
# # #         metrics = {}
# # #         for i, model in enumerate(self.models):
# # #             model.eval()
# # #             total_loss = 0.0
# # #             with torch.no_grad():
# # #                 for data, target in self.val_loader:
# # #                     data, target = data.to(self.device), target.to(self.device)
# # #                     output = model(data)
# # #                     if USE_PYTORCH3D:
# # #                         loss_cd, _ = pytorch3d_chamfer_distance(output.permute(0, 2, 1), target.permute(0, 2, 1))
# # #                     else:
# # #                         # Reshape to (batch_size, num_points, 3)
# # #                         output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
# # #                         target = target.permute(0, 2, 1)  # [batch_size, num_points, 3]
# # #                         loss_cd = 0.0
# # #                         for j in range(output.size(0)):
# # #                             loss_cd += chamfer_distance_cpu(output[j], target[j])
# # #                         loss_cd = loss_cd / output.size(0)
# # #                     total_loss += loss_cd.item()
# # #             average_loss = total_loss / len(self.val_loader)
# # #             metrics[f'validation_loss_node_{i}'] = average_loss
# # #             print(f"Validation Loss for node {i}: {average_loss}")
# # #         return metrics

# # #     def update_graph(self):
# # #         """
# # #         Update the communication graph if needed.
# # #         """
# # #         # Implement any dynamic graph updates if required
# # #         print("Updating communication graph...")
# # #         # Example: No dynamic updates; keep the graph static
# # #         pass

# # # # Define DiNNO Optimizer Class
# # # class DiNNO:
# # #     def __init__(self, ddl_problem, device, conf):
# # #         self.pr = ddl_problem
# # #         self.conf = conf

# # #         # Initialize dual variables
# # #         self.duals = {
# # #             i: torch.zeros((self.pr.n), device=device)
# # #             for i in range(self.pr.N)
# # #         }

# # #         # Initialize penalty parameter rho
# # #         self.rho = self.conf["rho_init"]
# # #         self.rho_scaling = self.conf["rho_scaling"]

# # #         # Learning rate scheduling
# # #         if self.conf["lr_decay_type"] == "constant":
# # #             self.primal_lr = self.conf["primal_lr_start"] * torch.ones(
# # #                 self.conf["outer_iterations"]
# # #             )
# # #         elif self.conf["lr_decay_type"] == "linear":
# # #             self.primal_lr = np.linspace(
# # #                 self.conf["primal_lr_start"],
# # #                 self.conf["primal_lr_finish"],
# # #                 self.conf["outer_iterations"],
# # #             )
# # #         elif self.conf["lr_decay_type"] == "log":
# # #             self.primal_lr = np.logspace(
# # #                 math.log10(self.conf["primal_lr_start"]),
# # #                 math.log10(self.conf["primal_lr_finish"]),
# # #                 self.conf["outer_iterations"],
# # #             )
# # #         else:
# # #             raise ValueError("Unknown primal learning rate decay type.")

# # #         self.pits = self.conf["primal_iterations"]

# # #         # Initialize optimizers
# # #         if self.conf["persistant_primal_opt"]:
# # #             self.opts = {}
# # #             for i in range(self.pr.N):
# # #                 if self.conf["primal_optimizer"] == "adam":
# # #                     self.opts[i] = torch.optim.Adam(
# # #                         self.pr.models[i].parameters(), self.primal_lr[0]
# # #                     )
# # #                 elif self.conf["primal_optimizer"] == "sgd":
# # #                     self.opts[i] = torch.optim.SGD(
# # #                         self.pr.models[i].parameters(), self.primal_lr[0]
# # #                     )
# # #                 elif self.conf["primal_optimizer"] == "adamw":
# # #                     self.opts[i] = torch.optim.AdamW(
# # #                         self.pr.models[i].parameters(), self.primal_lr[0]
# # #                     )
# # #                 else:
# # #                     raise ValueError("DiNNO primal optimizer is unknown.")

# # #         # Initialize metrics storage
# # #         self.metrics = []  # List to store metrics per epoch

# # #         # Early Stopping parameters
# # #         self.best_loss = float('inf')
# # #         self.patience = 20  # Number of iterations to wait before stopping
# # #         self.counter = 0

# # #     def primal_update(self, i, th_reg, k):
# # #         if self.conf["persistant_primal_opt"]:
# # #             opt = self.opts[i]
# # #             # Update learning rate
# # #             for param_group in opt.param_groups:
# # #                 param_group['lr'] = self.primal_lr[k]
# # #         else:
# # #             if self.conf["primal_optimizer"] == "adam":
# # #                 opt = torch.optim.Adam(
# # #                     self.pr.models[i].parameters(), self.primal_lr[k]
# # #                 )
# # #             elif self.conf["primal_optimizer"] == "sgd":
# # #                 opt = torch.optim.SGD(
# # #                     self.pr.models[i].parameters(), self.primal_lr[k]
# # #                 )
# # #             elif self.conf["primal_optimizer"] == "adamw":
# # #                 opt = torch.optim.AdamW(
# # #                     self.pr.models[i].parameters(), self.primal_lr[k]
# # #                 )
# # #             else:
# # #                 raise ValueError("DiNNO primal optimizer is unknown.")

# # #         for _ in range(self.pits):
# # #             opt.zero_grad()

# # #             # Model pass on the batch
# # #             pred_loss = self.pr.local_batch_loss(i)

# # #             # Get the primal variable WITH the autodiff graph attached.
# # #             th = torch.nn.utils.parameters_to_vector(
# # #                 self.pr.models[i].parameters()
# # #             )

# # #             reg = torch.sum(
# # #                 torch.square(th - th_reg)
# # #             )

# # #             loss = pred_loss + torch.dot(th, self.duals[i]) + self.rho * reg
# # #             loss.backward()

# # #             # Gradient clipping to prevent exploding gradients
# # #             torch.nn.utils.clip_grad_norm_(self.pr.models[i].parameters(), max_norm=1.0)

# # #             opt.step()

# # #         return

# # #     def synchronize_weights(self):
# # #         """
# # #         Synchronize model weights with neighboring nodes by averaging.
# # #         """
# # #         for i in range(self.pr.N):
# # #             neighbors = list(self.pr.graph.neighbors(i))
# # #             if neighbors:
# # #                 # Collect weights from neighbors
# # #                 neighbor_weights = []
# # #                 for neighbor in neighbors:
# # #                     neighbor_weights.append(torch.nn.utils.parameters_to_vector(
# # #                         self.pr.models[neighbor].parameters()
# # #                     ))
# # #                 # Average the weights
# # #                 avg_weights = torch.mean(torch.stack(neighbor_weights), dim=0)
# # #                 # Get current model weights
# # #                 current_weights = torch.nn.utils.parameters_to_vector(
# # #                     self.pr.models[i].parameters()
# # #                 )
# # #                 # Update local model weights by averaging with neighbor's weights
# # #                 new_weights = (current_weights + avg_weights) / 2.0
# # #                 torch.nn.utils.vector_to_parameters(new_weights, self.pr.models[i].parameters())

# # #                 # Debugging: Print a confirmation
# # #                 if self.conf["verbose"]:
# # #                     print(f"Node {i} weights synchronized with neighbors.")

# # #     def train(self, profiler=None):
# # #         eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
# # #         oits = self.conf["outer_iterations"]
# # #         for k in range(oits):
# # #             if k % eval_every == 0 or k == oits - 1:
# # #                 # Evaluate metrics and append to self.metrics
# # #                 metrics = self.pr.evaluate_metrics(at_end=(k == oits - 1), iteration=k)
# # #                 self.metrics.append(metrics)  # Store metrics

# # #                 # Calculate average validation loss across all nodes
# # #                 current_loss = np.mean(list(metrics.values()))

# # #                 # Early Stopping Check
# # #                 if current_loss < self.best_loss:
# # #                     self.best_loss = current_loss
# # #                     self.counter = 0
# # #                     print(f"Iteration {k}: Validation loss improved to {current_loss:.6f}")
# # #                 else:
# # #                     self.counter += 1
# # #                     print(f"Iteration {k}: No improvement in validation loss.")
# # #                     if self.counter >= self.patience:
# # #                         print("Early stopping triggered.")
# # #                         break

# # #             # Log rho value
# # #             if self.conf["verbose"]:
# # #                 print(f"Iteration {k}, Rho: {self.rho}")

# # #             # Get the current primal variables
# # #             ths = {
# # #                 i: torch.nn.utils.parameters_to_vector(
# # #                     self.pr.models[i].parameters()
# # #                 )
# # #                 .clone()
# # #                 .detach()
# # #                 for i in range(self.pr.N)
# # #             }

# # #             # Update the penalty parameter
# # #             self.rho *= self.rho_scaling

# # #             # Update the communication graph
# # #             self.pr.update_graph()

# # #             # Per node updates
# # #             for i in range(self.pr.N):
# # #                 neighs = list(self.pr.graph.neighbors(i))
# # #                 if neighs:
# # #                     thj = torch.stack([ths[j] for j in neighs])
# # #                     thj_mean = torch.mean(thj, dim=0)
# # #                 else:
# # #                     thj_mean = ths[i]

# # #                 self.duals[i] += self.rho * (ths[i] - thj_mean)
# # #                 th_reg = thj_mean
# # #                 self.primal_update(i, th_reg, k)

# # #             # Synchronize weights after each outer iteration
# # #             self.synchronize_weights()

# # #             if profiler is not None:
# # #                 profiler.step()

# # #         return

# # # # Define Reconstruction and Alignment Functions
# # # def align_point_clouds(source_pcd, target_pcd, threshold=0.02):
# # #     """
# # #     Align source_pcd to target_pcd using ICP.

# # #     Args:
# # #         source_pcd: Open3D PointCloud object to be aligned.
# # #         target_pcd: Open3D PointCloud object to align to.
# # #         threshold: Distance threshold for ICP.

# # #     Returns:
# # #         Aligned source_pcd.
# # #     """
# # #     transformation = o3d.pipelines.registration.registration_icp(
# # #         source_pcd, target_pcd, threshold, np.identity(4),
# # #         o3d.pipelines.registration.TransformationEstimationPointToPoint()
# # #     ).transformation
# # #     source_pcd.transform(transformation)
# # #     return source_pcd

# # # def reconstruct_and_align_map(ddl_problem, device):
# # #     """
# # #     Reconstruct the entire map by aggregating and aligning local reconstructions from all nodes.

# # #     Args:
# # #         ddl_problem: Instance of DDLProblem containing models and data loaders.
# # #         device: Torch device.

# # #     Returns:
# # #         global_map: Open3D PointCloud object representing the global map.
# # #     """
# # #     reconstructed_pcds = []
# # #     for i in range(ddl_problem.N):
# # #         model = ddl_problem.models[i].to(device)
# # #         model.eval()
# # #         all_reconstructions = []
# # #         with torch.no_grad():
# # #             for data, _ in ddl_problem.train_loaders[i]:
# # #                 data = data.to(device)
# # #                 output = model(data)
# # #                 output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
# # #                 all_reconstructions.append(output.cpu().numpy())
# # #         reconstructed_points = np.concatenate(all_reconstructions, axis=0).reshape(-1, 3)
# # #         pcd = o3d.geometry.PointCloud()
# # #         pcd.points = o3d.utility.Vector3dVector(reconstructed_points)
# # #         pcd = pcd.voxel_down_sample(voxel_size=0.05)  # Optional: Downsample for efficiency
# # #         reconstructed_pcds.append(pcd)

# # #     # Initialize global map with the first node's reconstruction
# # #     global_map = reconstructed_pcds[0]

# # #     for pcd in reconstructed_pcds[1:]:
# # #         aligned_pcd = align_point_clouds(pcd, global_map)
# # #         global_map += aligned_pcd
# # #         global_map = global_map.voxel_down_sample(voxel_size=0.05)  # Optional: Downsample after merging

# # #     return global_map

# # # # Main Training Function
# # # def train_dinno(ddl_problem, loss, val_set, graph, device, conf):
# # #     # Define the DiNNO optimizer
# # #     optimizer = DiNNO(ddl_problem, device, conf)

# # #     # Start training
# # #     optimizer.train()

# # #     return optimizer.metrics

# # # # Function to read KITTI Velodyne .bin file
# # # def read_kitti_bin(file_path):
# # #     """
# # #     Read a KITTI Velodyne .bin file and extract 3D point cloud (x, y, z).

# # #     Args:
# # #         file_path: Path to the .bin file.

# # #     Returns:
# # #         numpy array of shape (N, 3), where N is the number of points.
# # #     """
# # #     points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
# # #     return points[:, :3]  # Extract only x, y, z coordinates

# # # # Main Execution Block
# # # if __name__ == "__main__":
# # #     # Configuration
# # #     conf = {
# # #         "output_metadir": "E:/Documents/Masters_Courses/ESE 5460/final_project_env/DL_Project/output",
# # #         "name": "3d_map_DiNNO",
# # #         "epochs": 100,  # Corresponds to 'outer_iterations' in DiNNO
# # #         "verbose": True,
# # #         "graph": {
# # #             "type": "cycle",  # Options: "fully_connected", "cycle", "ring", "star", "erdos_renyi"
# # #             "num_nodes": 2,  # Decreased from 10 to 2
# # #             "p": 0.3,
# # #             "gen_attempts": 100
# # #         },
# # #         "train_batch_size": 8,
# # #         "val_batch_size": 8,
# # #         "data_dir": "E:/Documents/Masters_Courses/ESE 5460/final_project_env/DL_Project/2011_09_28_drive_0035_sync/2011_09_28/2011_09_28_drive_0035_sync/velodyne_points/data",
# # #         "model": {
# # #             "in_channels": 3,  # Updated for PointNet
# # #             "out_channels": 3,
# # #             "init_features": 3,
# # #             "kernel_size": 1,
# # #             "linear_width": 64
# # #         },
# # #         "loss": "Chamfer",  # Options: "Chamfer"
# # #         "use_cuda": torch.cuda.is_available(),
# # #         "individual_training": {
# # #             "train_solo": False,
# # #             "optimizer": "adam",
# # #             "lr": 0.0005,  # Adjusted learning rate
# # #             "verbose": True
# # #         },
# # #         # DiNNO Specific Hyperparameters
# # #         "rho_init": 0.05,               # Adjusted from 0.1
# # #         "rho_scaling": 1.05,            # Adjusted from 1.1
# # #         "lr_decay_type": "linear",      # Changed from "constant" to "linear"
# # #         "primal_lr_start": 0.0005,      # Adjusted from 0.001
# # #         "primal_lr_finish": 0.00005,    # Adjusted from 0.0001
# # #         "outer_iterations": 100,        # Number of outer iterations
# # #         "primal_iterations": 20,        # Increased from 10
# # #         "persistant_primal_opt": True,  # Use persistent primal optimizers
# # #         "primal_optimizer": "adam",     # Type of primal optimizer: 'adam', 'sgd', 'adamw'
# # #         "metrics_config": {             # Metrics configuration (if used)
# # #             "evaluate_frequency": 1     # Evaluate metrics every iteration
# # #         },
# # #         "device": DEVICE.type,           # 'cuda' or 'cpu'
# # #         "num_points": 1024               # Number of points per point cloud
# # #     }

# # #     # Create output directory
# # #     if not os.path.exists(conf["output_metadir"]):
# # #         os.makedirs(conf["output_metadir"], exist_ok=True)

# # #     # Load point cloud data from KITTI .bin files
# # #     all_point_clouds = []
# # #     for file_name in sorted(os.listdir(conf["data_dir"])):
# # #         if file_name.endswith('.bin'):
# # #             file_path = os.path.join(conf["data_dir"], file_name)
# # #             points = read_kitti_bin(file_path)
# # #             all_point_clouds.append(points)

# # #     print(f"Total point clouds loaded: {len(all_point_clouds)}")

# # #     # Adjust the number of nodes to 2
# # #     num_nodes = conf["graph"]["num_nodes"]

# # #     # Split point clouds among nodes
# # #     node_point_clouds = [[] for _ in range(num_nodes)]
# # #     for idx, point_cloud in enumerate(all_point_clouds):
# # #         node_idx = idx % num_nodes  # Distribute in a round-robin fashion
# # #         node_point_clouds[node_idx].append(point_cloud)

# # #     # Verify the distribution
# # #     for i in range(num_nodes):
# # #         print(f"Node {i} has {len(node_point_clouds[i])} point clouds.")

# # #     # Create validation set (use 10% of each node's data)
# # #     val_point_clouds = []
# # #     for i in range(num_nodes):
# # #         num_val_samples = max(1, len(node_point_clouds[i]) // 10)
# # #         val_point_clouds.extend(node_point_clouds[i][:num_val_samples])
# # #         node_point_clouds[i] = node_point_clouds[i][num_val_samples:]  # Remove validation data from training data

# # #     val_set = PointCloudDataset(
# # #         val_point_clouds,
# # #         num_points=conf["num_points"],
# # #         augment=False
# # #     )
# # #     print(f"Validation set size: {len(val_set)}")

# # #     # Create training subsets for each node
# # #     train_subsets = []
# # #     for i in range(num_nodes):
# # #         point_cloud_list = node_point_clouds[i]
# # #         dataset = PointCloudDataset(
# # #             point_cloud_list,
# # #             num_points=conf["num_points"],
# # #             augment=True
# # #         )
# # #         train_subsets.append(dataset)
# # #         print(f"Node {i} has {len(dataset)} training samples.")

# # #     # Create base models for each node
# # #     models = [PointNetAutoencoder(num_points=conf["num_points"], num_groups=32).to(DEVICE) for _ in range(num_nodes)]
# # #     print(f"Created {num_nodes} PointNetAutoencoders.")

# # #     # Verify Model Dtypes
# # #     for idx, model in enumerate(models):
# # #         for name, param in model.named_parameters():
# # #             print(f"Model {idx}, Parameter {name}, dtype: {param.dtype}")

# # #     # Create DDLProblem instance
# # #     ddl_problem = DDLProblem(models=models, N=num_nodes, conf=conf, train_subsets=train_subsets, val_set=val_set)
# # #     print("DDLProblem instance created.")

# # #     # Train using DiNNO
# # #     if conf["individual_training"]["train_solo"]:
# # #         print("Performing individual training...")
# # #         # Implement individual training logic here if needed
# # #         raise NotImplementedError("Individual training not implemented.")
# # #     else:
# # #         try:
# # #             metrics = train_dinno(ddl_problem, None, val_set, None, DEVICE, conf)
# # #         except Exception as e:
# # #             print(f"An error occurred during training: {e}")
# # #             metrics = None

# # #         if metrics is not None:
# # #             # Save metrics and models
# # #             torch.save(metrics, os.path.join(conf["output_metadir"], "dinno_metrics.pt"))
# # #             for idx, model in enumerate(ddl_problem.models):
# # #                 torch.save(model.state_dict(), os.path.join(conf["output_metadir"], f"dinno_trained_model_{idx}.pth"))
# # #             print("Training complete. Metrics and models saved.")

# # #             # Reconstruct and visualize the whole map with alignment
# # #             print("Reconstructing the global map...")
# # #             global_map = reconstruct_and_align_map(ddl_problem, DEVICE)
# # #             print("Global map reconstructed.")

# # #             # Visualize the global map
# # #             o3d.visualization.draw_geometries([global_map], window_name="Reconstructed Global Map")

# # #             # Save the global map
# # #             o3d.io.write_point_cloud(os.path.join(conf["output_metadir"], "reconstructed_global_map.pcd"), global_map)
# # #             print("Reconstructed global map saved.")

# # #     print("Script execution completed.")


# # import os
# # import sys
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.utils.data import Dataset, DataLoader
# # import networkx as nx
# # import math
# # from datetime import datetime
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# # # Check for CUDA availability
# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print(f"Device is set to {'GPU' if DEVICE.type == 'cuda' else 'CPU'}")

# # # Define PointNet-Based Autoencoder with GroupNorm
# # class PointNetAutoencoder(nn.Module):
# #     def __init__(self, num_points=1024, num_groups=32):
# #         super(PointNetAutoencoder, self).__init__()
# #         self.num_points = num_points
# #         self.num_groups = num_groups  # Number of groups for GroupNorm

# #         # Encoder
# #         self.conv1 = nn.Conv1d(3, 64, 1)
# #         self.gn1 = nn.GroupNorm(num_groups=self.num_groups, num_channels=64)
# #         self.conv2 = nn.Conv1d(64, 128, 1)
# #         self.gn2 = nn.GroupNorm(num_groups=self.num_groups, num_channels=128)
# #         self.conv3 = nn.Conv1d(128, 1024, 1)
# #         self.gn3 = nn.GroupNorm(num_groups=self.num_groups, num_channels=1024)
# #         self.fc1 = nn.Linear(1024, 512)
# #         self.gn4 = nn.GroupNorm(num_groups=self.num_groups, num_channels=512)
# #         self.fc2 = nn.Linear(512, 256)
# #         self.gn5 = nn.GroupNorm(num_groups=self.num_groups, num_channels=256)
        
# #         # Decoder
# #         self.fc3 = nn.Linear(256, 512)
# #         self.gn6 = nn.GroupNorm(num_groups=self.num_groups, num_channels=512)
# #         self.fc4 = nn.Linear(512, 3 * self.num_points)

# #     def forward(self, x):
# #         # Encoder
# #         x = F.relu(self.gn1(self.conv1(x)))  # [batch_size, 64, num_points]
# #         x = F.relu(self.gn2(self.conv2(x)))  # [batch_size, 128, num_points]
# #         x = F.relu(self.gn3(self.conv3(x)))  # [batch_size, 1024, num_points]
# #         x = torch.max(x, 2, keepdim=True)[0]  # [batch_size, 1024, 1]
# #         x = x.view(-1, 1024)  # [batch_size, 1024]
# #         x = F.relu(self.gn4(self.fc1(x)))  # [batch_size, 512]
# #         x = F.relu(self.gn5(self.fc2(x)))  # [batch_size, 256]
        
# #         # Decoder
# #         x = F.relu(self.gn6(self.fc3(x)))  # [batch_size, 512]
# #         x = self.fc4(x)  # [batch_size, 3 * num_points]
# #         x = x.view(-1, 3, self.num_points)  # [batch_size, 3, num_points]
# #         return x

# # # Define Custom Dataset for Point Clouds
# # class PointCloudDataset(Dataset):
# #     def __init__(self, point_cloud_list, num_points=1024, augment=False):
# #         """
# #         Args:
# #             point_cloud_list: List of numpy arrays, each of shape (N_i, 3).
# #             num_points: Number of points to sample from each point cloud.
# #             augment: Whether to apply data augmentation.
# #         """
# #         self.point_cloud_list = point_cloud_list
# #         self.augment = augment
# #         self.num_points = num_points

# #     def __len__(self):
# #         return len(self.point_cloud_list)

# #     def __getitem__(self, idx):
# #         point_cloud = self.point_cloud_list[idx]
# #         num_points_in_cloud = point_cloud.shape[0]

# #         if num_points_in_cloud >= self.num_points:
# #             # Randomly sample num_points
# #             indices = np.random.choice(num_points_in_cloud, self.num_points, replace=False)
# #             data = point_cloud[indices]
# #         else:
# #             # If fewer points, pad with random points
# #             pad_size = self.num_points - num_points_in_cloud
# #             pad = np.random.normal(0, 0.001, size=(pad_size, 3))
# #             data = np.concatenate([point_cloud, pad], axis=0)

# #         if self.augment:
# #             # Apply random rotation around Z-axis
# #             theta = np.random.uniform(0, 2 * np.pi)
# #             rotation_matrix = np.array([
# #                 [np.cos(theta), -np.sin(theta), 0],
# #                 [np.sin(theta),  np.cos(theta), 0],
# #                 [0,              0,             1]
# #             ])
# #             data = data @ rotation_matrix.T

# #         # Normalize per point cloud
# #         mean = np.mean(data, axis=0)
# #         data = data - mean  # Center the point cloud

# #         data = torch.tensor(data, dtype=torch.float32)
# #         data = data.T  # Transpose to shape (3, num_points)
# #         return data, data  # Return (input, target)

# # # Placeholder for Graph Generation Utility
# # def generate_from_conf(graph_conf):
# #     """
# #     Generates a communication graph based on the configuration.

# #     Args:
# #         graph_conf: Dictionary containing graph configuration.

# #     Returns:
# #         N: Number of nodes.
# #         graph: NetworkX graph.
# #     """
# #     graph_type = graph_conf.get("type", "fully_connected")
# #     num_nodes = graph_conf.get("num_nodes", 10)
# #     p = graph_conf.get("p", 0.3)
# #     gen_attempts = graph_conf.get("gen_attempts", 100)

# #     if graph_type == "fully_connected":
# #         graph = nx.complete_graph(num_nodes)
# #     elif graph_type == "cycle":
# #         graph = nx.cycle_graph(num_nodes)
# #     elif graph_type == "ring":
# #         graph = nx.cycle_graph(num_nodes)
# #     elif graph_type == "star":
# #         graph = nx.star_graph(num_nodes - 1)
# #     elif graph_type == "erdos_renyi":
# #         graph = nx.erdos_renyi_graph(num_nodes, p, seed=None, directed=False)
# #         attempts = 0
# #         while not nx.is_connected(graph) and attempts < gen_attempts:
# #             graph = nx.erdos_renyi_graph(num_nodes, p, seed=None, directed=False)
# #             attempts += 1
# #         if not nx.is_connected(graph):
# #             raise ValueError("Failed to generate a connected Erdos-Renyi graph.")
# #     else:
# #         raise ValueError(f"Unknown graph type: {graph_type}")

# #     return num_nodes, graph

# # # Define DDLProblem Class
# # class DDLProblem:
# #     def __init__(self, models, N, conf, train_subsets, val_set):
# #         self.models = models
# #         self.N = N
# #         self.conf = conf
# #         # Calculate the total number of parameters in the first model
# #         self.n = torch.numel(torch.nn.utils.parameters_to_vector(self.models[0].parameters()))
# #         # Initialize communication graph
# #         self.graph = generate_from_conf(conf["graph"])[1]
# #         # Initialize data loaders for each node with drop_last=False to retain all data
# #         self.train_loaders = [
# #             DataLoader(
# #                 train_subsets[i],
# #                 batch_size=conf["train_batch_size"],
# #                 shuffle=True,
# #                 drop_last=False  # Ensures all data is processed
# #             )
# #             for i in range(N)
# #         ]
# #         self.val_loader = DataLoader(
# #             val_set,
# #             batch_size=conf["val_batch_size"],
# #             shuffle=False,
# #             drop_last=False  # Ensures all validation data is processed
# #         )
# #         # Assign device
# #         self.device = torch.device(conf["device"])
# #         # Initialize iterators for each train loader
# #         self.train_iters = [iter(loader) for loader in self.train_loaders]

# #     def local_batch_loss(self, i):
# #         """
# #         Compute the local batch loss for node i using Mean Squared Error loss.
# #         """
# #         model = self.models[i].to(self.device)
# #         model.train()
# #         try:
# #             data, target = next(self.train_iters[i])
# #         except StopIteration:
# #             # Restart the loader if the iterator is exhausted
# #             self.train_iters[i] = iter(self.train_loaders[i])
# #             data, target = next(self.train_iters[i])
# #         data, target = data.to(self.device), target.to(self.device)
# #         output = model(data)
# #         # Use Mean Squared Error loss
# #         loss_mse = F.mse_loss(output, target)
# #         return loss_mse

# #     def evaluate_metrics(self, at_end=False, iteration=0):
# #         """
# #         Evaluate and return metrics such as validation loss using Mean Squared Error loss.
# #         """
# #         metrics = {}
# #         for i, model in enumerate(self.models):
# #             model.eval()
# #             total_loss = 0.0
# #             with torch.no_grad():
# #                 for data, target in self.val_loader:
# #                     data, target = data.to(self.device), target.to(self.device)
# #                     output = model(data)
# #                     loss_mse = F.mse_loss(output, target)
# #                     total_loss += loss_mse.item()
# #             average_loss = total_loss / len(self.val_loader)
# #             metrics[f'validation_loss_node_{i}'] = average_loss
# #             print(f"Validation Loss for node {i}: {average_loss}")
# #         return metrics

# #     def update_graph(self):
# #         """
# #         Update the communication graph if needed.
# #         """
# #         # Implement any dynamic graph updates if required
# #         print("Updating communication graph...")
# #         # Example: No dynamic updates; keep the graph static
# #         pass

# # # Define DiNNO Optimizer Class
# # class DiNNO:
# #     def __init__(self, ddl_problem, device, conf):
# #         self.pr = ddl_problem
# #         self.conf = conf

# #         # Initialize dual variables
# #         self.duals = {
# #             i: torch.zeros((self.pr.n), device=device)
# #             for i in range(self.pr.N)
# #         }

# #         # Initialize penalty parameter rho
# #         self.rho = self.conf["rho_init"]
# #         self.rho_scaling = self.conf["rho_scaling"]

# #         # Learning rate scheduling
# #         if self.conf["lr_decay_type"] == "constant":
# #             self.primal_lr = self.conf["primal_lr_start"] * torch.ones(
# #                 self.conf["outer_iterations"]
# #             )
# #         elif self.conf["lr_decay_type"] == "linear":
# #             self.primal_lr = np.linspace(
# #                 self.conf["primal_lr_start"],
# #                 self.conf["primal_lr_finish"],
# #                 self.conf["outer_iterations"],
# #             )
# #         elif self.conf["lr_decay_type"] == "log":
# #             self.primal_lr = np.logspace(
# #                 math.log10(self.conf["primal_lr_start"]),
# #                 math.log10(self.conf["primal_lr_finish"]),
# #                 self.conf["outer_iterations"],
# #             )
# #         else:
# #             raise ValueError("Unknown primal learning rate decay type.")

# #         self.pits = self.conf["primal_iterations"]

# #         # Initialize optimizers
# #         if self.conf["persistant_primal_opt"]:
# #             self.opts = {}
# #             for i in range(self.pr.N):
# #                 if self.conf["primal_optimizer"] == "adam":
# #                     self.opts[i] = torch.optim.Adam(
# #                         self.pr.models[i].parameters(), self.primal_lr[0]
# #                     )
# #                 elif self.conf["primal_optimizer"] == "sgd":
# #                     self.opts[i] = torch.optim.SGD(
# #                         self.pr.models[i].parameters(), self.primal_lr[0]
# #                     )
# #                 elif self.conf["primal_optimizer"] == "adamw":
# #                     self.opts[i] = torch.optim.AdamW(
# #                         self.pr.models[i].parameters(), self.primal_lr[0]
# #                     )
# #                 else:
# #                     raise ValueError("DiNNO primal optimizer is unknown.")

# #         # Initialize metrics storage
# #         self.metrics = []  # List to store metrics per epoch

# #         # Early Stopping parameters
# #         self.best_loss = float('inf')
# #         self.patience = 20  # Number of iterations to wait before stopping
# #         self.counter = 0

# #     def primal_update(self, i, th_reg, k):
# #         if self.conf["persistant_primal_opt"]:
# #             opt = self.opts[i]
# #             # Update learning rate
# #             for param_group in opt.param_groups:
# #                 param_group['lr'] = self.primal_lr[k]
# #         else:
# #             if self.conf["primal_optimizer"] == "adam":
# #                 opt = torch.optim.Adam(
# #                     self.pr.models[i].parameters(), self.primal_lr[k]
# #                 )
# #             elif self.conf["primal_optimizer"] == "sgd":
# #                 opt = torch.optim.SGD(
# #                     self.pr.models[i].parameters(), self.primal_lr[k]
# #                 )
# #             elif self.conf["primal_optimizer"] == "adamw":
# #                 opt = torch.optim.AdamW(
# #                     self.pr.models[i].parameters(), self.primal_lr[k]
# #                 )
# #             else:
# #                 raise ValueError("DiNNO primal optimizer is unknown.")

# #         for _ in range(self.pits):
# #             opt.zero_grad()

# #             # Model pass on the batch
# #             pred_loss = self.pr.local_batch_loss(i)

# #             # Get the primal variable WITH the autodiff graph attached.
# #             th = torch.nn.utils.parameters_to_vector(
# #                 self.pr.models[i].parameters()
# #             )

# #             reg = torch.sum(
# #                 torch.square(th - th_reg)
# #             )

# #             loss = pred_loss + torch.dot(th, self.duals[i]) + self.rho * reg
# #             loss.backward()

# #             # Gradient clipping to prevent exploding gradients
# #             torch.nn.utils.clip_grad_norm_(self.pr.models[i].parameters(), max_norm=1.0)

# #             opt.step()

# #         return

# #     def synchronize_weights(self):
# #         """
# #         Synchronize model weights with neighboring nodes by averaging.
# #         """
# #         for i in range(self.pr.N):
# #             neighbors = list(self.pr.graph.neighbors(i))
# #             if neighbors:
# #                 # Collect weights from neighbors
# #                 neighbor_weights = []
# #                 for neighbor in neighbors:
# #                     neighbor_weights.append(torch.nn.utils.parameters_to_vector(
# #                         self.pr.models[neighbor].parameters()
# #                     ))
# #                 # Average the weights
# #                 avg_weights = torch.mean(torch.stack(neighbor_weights), dim=0)
# #                 # Get current model weights
# #                 current_weights = torch.nn.utils.parameters_to_vector(
# #                     self.pr.models[i].parameters()
# #                 )
# #                 # Update local model weights by averaging with neighbor's weights
# #                 new_weights = (current_weights + avg_weights) / 2.0
# #                 torch.nn.utils.vector_to_parameters(new_weights, self.pr.models[i].parameters())

# #                 # Debugging: Print a confirmation
# #                 if self.conf["verbose"]:
# #                     print(f"Node {i} weights synchronized with neighbors.")

# #     def train(self, profiler=None):
# #         eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
# #         oits = self.conf["outer_iterations"]
# #         for k in range(oits):
# #             if k % eval_every == 0 or k == oits - 1:
# #                 # Evaluate metrics and append to self.metrics
# #                 metrics = self.pr.evaluate_metrics(at_end=(k == oits - 1), iteration=k)
# #                 self.metrics.append(metrics)  # Store metrics

# #                 # Calculate average validation loss across all nodes
# #                 current_loss = np.mean(list(metrics.values()))

# #                 # Early Stopping Check
# #                 if current_loss < self.best_loss:
# #                     self.best_loss = current_loss
# #                     self.counter = 0
# #                     print(f"Iteration {k}: Validation loss improved to {current_loss:.6f}")
# #                 else:
# #                     self.counter += 1
# #                     print(f"Iteration {k}: No improvement in validation loss.")
# #                     if self.counter >= self.patience:
# #                         print("Early stopping triggered.")
# #                         break

# #             # Log rho value
# #             if self.conf["verbose"]:
# #                 print(f"Iteration {k}, Rho: {self.rho}")

# #             # Get the current primal variables
# #             ths = {
# #                 i: torch.nn.utils.parameters_to_vector(
# #                     self.pr.models[i].parameters()
# #                 )
# #                 .clone()
# #                 .detach()
# #                 for i in range(self.pr.N)
# #             }

# #             # Update the penalty parameter
# #             self.rho *= self.rho_scaling

# #             # Update the communication graph
# #             self.pr.update_graph()

# #             # Per node updates
# #             for i in range(self.pr.N):
# #                 neighs = list(self.pr.graph.neighbors(i))
# #                 if neighs:
# #                     thj = torch.stack([ths[j] for j in neighs])
# #                     thj_mean = torch.mean(thj, dim=0)
# #                 else:
# #                     thj_mean = ths[i]

# #                 self.duals[i] += self.rho * (ths[i] - thj_mean)
# #                 th_reg = thj_mean
# #                 self.primal_update(i, th_reg, k)

# #             # Synchronize weights after each outer iteration
# #             self.synchronize_weights()

# #             if profiler is not None:
# #                 profiler.step()

# #         return

# # # Define Reconstruction Function
# # def reconstruct_maps(ddl_problem, device, output_dir):
# #     """
# #     Reconstruct and save the point clouds from each node.

# #     Args:
# #         ddl_problem: Instance of DDLProblem containing models and data loaders.
# #         device: Torch device.
# #         output_dir: Directory to save the reconstructed maps.

# #     Returns:
# #         reconstructed_point_clouds: List of numpy arrays for each node.
# #     """
# #     reconstructed_point_clouds = []
# #     for i in range(ddl_problem.N):
# #         model = ddl_problem.models[i].to(device)
# #         model.eval()
# #         all_reconstructions = []
# #         with torch.no_grad():
# #             for data, _ in ddl_problem.train_loaders[i]:
# #                 data = data.to(device)
# #                 output = model(data)
# #                 output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
# #                 all_reconstructions.append(output.cpu().numpy())
# #         reconstructed_points = np.concatenate(all_reconstructions, axis=0).reshape(-1, 3)
# #         reconstructed_point_clouds.append(reconstructed_points)

# #         # Save the reconstructed points
# #         np.save(os.path.join(output_dir, f"reconstructed_map_node_{i}.npy"), reconstructed_points)
# #         print(f"Reconstructed map for node {i} saved.")

# #         # Visualize and save the plot
# #         fig = plt.figure()
# #         ax = fig.add_subplot(111, projection='3d')
# #         ax.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2], s=0.5)
# #         ax.set_title(f"Reconstructed Map Node {i}")
# #         plt.savefig(os.path.join(output_dir, f"reconstructed_map_node_{i}.png"))
# #         plt.close(fig)
# #         print(f"Reconstructed map for node {i} visualized and saved.")

# #     return reconstructed_point_clouds

# # # Main Training Function
# # def train_dinno(ddl_problem, device, conf):
# #     # Define the DiNNO optimizer
# #     optimizer = DiNNO(ddl_problem, device, conf)

# #     # Start training
# #     optimizer.train()

# #     return optimizer.metrics

# # # Function to read KITTI Velodyne .bin file
# # def read_kitti_bin(file_path):
# #     """
# #     Read a KITTI Velodyne .bin file and extract 3D point cloud (x, y, z).

# #     Args:
# #         file_path: Path to the .bin file.

# #     Returns:
# #         numpy array of shape (N, 3), where N is the number of points.
# #     """
# #     points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
# #     return points[:, :3]  # Extract only x, y, z coordinates

# # # Main Execution Block
# # if __name__ == "__main__":
# #     # Configuration
# #     conf = {
# #         "output_metadir": "E:/Documents/Masters_Courses/ESE 5460/final_project_env/DL_Project/output",
# #         "name": "3d_map_DiNNO",
# #         "epochs": 100,  # Corresponds to 'outer_iterations' in DiNNO
# #         "verbose": True,
# #         "graph": {
# #             "type": "cycle",  # Options: "fully_connected", "cycle", "ring", "star", "erdos_renyi"
# #             "num_nodes": 2,  # Adjusted number of nodes
# #             "p": 0.3,
# #             "gen_attempts": 100
# #         },
# #         "train_batch_size": 8,
# #         "val_batch_size": 8,
# #         "data_dir": "E:/Documents/Masters_Courses/ESE 5460/final_project_env/DL_Project/2011_09_28_drive_0035_sync/2011_09_28/2011_09_28_drive_0035_sync/velodyne_points/data",
# #         "model": {
# #             "in_channels": 3,  # Updated for PointNet
# #             "out_channels": 3,
# #             "init_features": 3,
# #             "kernel_size": 1,
# #             "linear_width": 64
# #         },
# #         "loss": "MSE",  # Changed to Mean Squared Error
# #         "use_cuda": torch.cuda.is_available(),
# #         "individual_training": {
# #             "train_solo": False,
# #             "optimizer": "adam",
# #             "lr": 0.0005,  # Adjusted learning rate
# #             "verbose": True
# #         },
# #         # DiNNO Specific Hyperparameters
# #         "rho_init": 0.05,               # Adjusted from 0.1
# #         "rho_scaling": 1.05,            # Adjusted from 1.1
# #         "lr_decay_type": "linear",      # Changed from "constant" to "linear"
# #         "primal_lr_start": 0.0005,      # Adjusted from 0.001
# #         "primal_lr_finish": 0.00005,    # Adjusted from 0.0001
# #         "outer_iterations": 100,        # Number of outer iterations
# #         "primal_iterations": 20,        # Increased from 10
# #         "persistant_primal_opt": True,  # Use persistent primal optimizers
# #         "primal_optimizer": "adam",     # Type of primal optimizer: 'adam', 'sgd', 'adamw'
# #         "metrics_config": {             # Metrics configuration (if used)
# #             "evaluate_frequency": 1     # Evaluate metrics every iteration
# #         },
# #         "device": DEVICE.type,           # 'cuda' or 'cpu'
# #         "num_points": 1024               # Number of points per point cloud
# #     }

# #     # Create output directory
# #     if not os.path.exists(conf["output_metadir"]):
# #         os.makedirs(conf["output_metadir"], exist_ok=True)

# #     # Create experiment output directory
# #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# #     output_dir = os.path.join(conf["output_metadir"], f"{conf['name']}_{timestamp}")
# #     os.makedirs(output_dir, exist_ok=True)

# #     # Load point cloud data from KITTI .bin files
# #     all_point_clouds = []
# #     for file_name in sorted(os.listdir(conf["data_dir"])):
# #         if file_name.endswith('.bin'):
# #             file_path = os.path.join(conf["data_dir"], file_name)
# #             points = read_kitti_bin(file_path)
# #             all_point_clouds.append(points)

# #     print(f"Total point clouds loaded: {len(all_point_clouds)}")

# #     # Adjust the number of nodes
# #     num_nodes = conf["graph"]["num_nodes"]

# #     # Split point clouds among nodes
# #     node_point_clouds = [[] for _ in range(num_nodes)]
# #     for idx, point_cloud in enumerate(all_point_clouds):
# #         node_idx = idx % num_nodes  # Distribute in a round-robin fashion
# #         node_point_clouds[node_idx].append(point_cloud)

# #     # Verify the distribution
# #     for i in range(num_nodes):
# #         print(f"Node {i} has {len(node_point_clouds[i])} point clouds.")

# #     # Create validation set (use 10% of each node's data)
# #     val_point_clouds = []
# #     for i in range(num_nodes):
# #         num_val_samples = max(1, len(node_point_clouds[i]) // 10)
# #         val_point_clouds.extend(node_point_clouds[i][:num_val_samples])
# #         node_point_clouds[i] = node_point_clouds[i][num_val_samples:]  # Remove validation data from training data

# #     val_set = PointCloudDataset(
# #         val_point_clouds,
# #         num_points=conf["num_points"],
# #         augment=False
# #     )
# #     print(f"Validation set size: {len(val_set)}")

# #     # Create training subsets for each node
# #     train_subsets = []
# #     for i in range(num_nodes):
# #         point_cloud_list = node_point_clouds[i]
# #         dataset = PointCloudDataset(
# #             point_cloud_list,
# #             num_points=conf["num_points"],
# #             augment=True
# #         )
# #         train_subsets.append(dataset)
# #         print(f"Node {i} has {len(dataset)} training samples.")

# #     # Create base models for each node
# #     models = [PointNetAutoencoder(num_points=conf["num_points"], num_groups=32).to(DEVICE) for _ in range(num_nodes)]
# #     print(f"Created {num_nodes} PointNetAutoencoders.")

# #     # Verify Model Dtypes
# #     for idx, model in enumerate(models):
# #         for name, param in model.named_parameters():
# #             print(f"Model {idx}, Parameter {name}, dtype: {param.dtype}")

# #     # Create DDLProblem instance
# #     ddl_problem = DDLProblem(models=models, N=num_nodes, conf=conf, train_subsets=train_subsets, val_set=val_set)
# #     print("DDLProblem instance created.")

# #     # Train using DiNNO
# #     if conf["individual_training"]["train_solo"]:
# #         print("Performing individual training...")
# #         # Implement individual training logic here if needed
# #         raise NotImplementedError("Individual training not implemented.")
# #     else:
# #         try:
# #             metrics = train_dinno(ddl_problem, DEVICE, conf)
# #         except Exception as e:
# #             print(f"An error occurred during training: {e}")
# #             metrics = None

# #         if metrics is not None:
# #             # Save metrics and models
# #             torch.save(metrics, os.path.join(output_dir, "dinno_metrics.pt"))
# #             for idx, model in enumerate(ddl_problem.models):
# #                 torch.save(model.state_dict(), os.path.join(output_dir, f"dinno_trained_model_{idx}.pth"))
# #             print("Training complete. Metrics and models saved.")

# #             # Reconstruct and save the maps from each node
# #             print("Reconstructing maps from each node...")
# #             reconstructed_point_clouds = reconstruct_maps(ddl_problem, DEVICE, output_dir)
# #             print("Reconstruction complete.")

# #             # Optionally, combine and visualize the global map
# #             print("Combining and visualizing the global map...")
# #             global_map_points = np.vstack(reconstructed_point_clouds)
# #             fig = plt.figure()
# #             ax = fig.add_subplot(111, projection='3d')
# #             ax.scatter(global_map_points[:, 0], global_map_points[:, 1], global_map_points[:, 2], s=0.5)
# #             ax.set_title("Combined Global Map")
# #             plt.savefig(os.path.join(output_dir, "combined_global_map.png"))
# #             plt.close(fig)
# #             print("Combined global map visualized and saved.")

# #     print("Script execution completed.")


# import os
# import sys
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import networkx as nx
# import math
# from datetime import datetime
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# # Check for CUDA availability
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device is set to {'GPU' if DEVICE.type == 'cuda' else 'CPU'}")

# # Define PointNet-Based Autoencoder with GroupNorm
# class PointNetAutoencoder(nn.Module):
#     def __init__(self, num_points=1024, num_groups=32):
#         super(PointNetAutoencoder, self).__init__()
#         self.num_points = num_points
#         self.num_groups = num_groups  # Number of groups for GroupNorm

#         # Encoder
#         self.conv1 = nn.Conv1d(3, 64, 1)
#         self.gn1 = nn.GroupNorm(num_groups=self.num_groups, num_channels=64)
#         self.conv2 = nn.Conv1d(64, 128, 1)
#         self.gn2 = nn.GroupNorm(num_groups=self.num_groups, num_channels=128)
#         self.conv3 = nn.Conv1d(128, 1024, 1)
#         self.gn3 = nn.GroupNorm(num_groups=self.num_groups, num_channels=1024)
#         self.fc1 = nn.Linear(1024, 512)
#         self.gn4 = nn.GroupNorm(num_groups=self.num_groups, num_channels=512)
#         self.fc2 = nn.Linear(512, 256)
#         self.gn5 = nn.GroupNorm(num_groups=self.num_groups, num_channels=256)
        
#         # Decoder
#         self.fc3 = nn.Linear(256, 512)
#         self.gn6 = nn.GroupNorm(num_groups=self.num_groups, num_channels=512)
#         self.fc4 = nn.Linear(512, 3 * self.num_points)

#     def forward(self, x):
#         # Encoder
#         x = F.relu(self.gn1(self.conv1(x)))  # [batch_size, 64, num_points]
#         x = F.relu(self.gn2(self.conv2(x)))  # [batch_size, 128, num_points]
#         x = F.relu(self.gn3(self.conv3(x)))  # [batch_size, 1024, num_points]
#         x = torch.max(x, 2, keepdim=True)[0]  # [batch_size, 1024, 1]
#         x = x.view(-1, 1024)  # [batch_size, 1024]
#         x = F.relu(self.gn4(self.fc1(x)))  # [batch_size, 512]
#         x = F.relu(self.gn5(self.fc2(x)))  # [batch_size, 256]
        
#         # Decoder
#         x = F.relu(self.gn6(self.fc3(x)))  # [batch_size, 512]
#         x = self.fc4(x)  # [batch_size, 3 * self.num_points]
#         x = x.view(-1, 3, self.num_points)  # [batch_size, 3, num_points]
#         return x

# # Define Custom Dataset for Point Clouds
# class PointCloudDataset(Dataset):
#     def __init__(self, point_cloud_list, num_points=1024, augment=False):
#         """
#         Args:
#             point_cloud_list: List of numpy arrays, each of shape (N_i, 3).
#             num_points: Number of points to sample from each point cloud.
#             augment: Whether to apply data augmentation.
#         """
#         self.point_cloud_list = point_cloud_list
#         self.augment = augment
#         self.num_points = num_points

#     def __len__(self):
#         return len(self.point_cloud_list)

#     def __getitem__(self, idx):
#         point_cloud = self.point_cloud_list[idx]
#         num_points_in_cloud = point_cloud.shape[0]

#         if num_points_in_cloud >= self.num_points:
#             # Randomly sample num_points
#             indices = np.random.choice(num_points_in_cloud, self.num_points, replace=False)
#             data = point_cloud[indices]
#         else:
#             # If fewer points, pad with random points
#             pad_size = self.num_points - num_points_in_cloud
#             pad = np.random.normal(0, 0.001, size=(pad_size, 3))
#             data = np.concatenate([point_cloud, pad], axis=0)

#         if self.augment:
#             # Apply random rotation around Z-axis
#             theta = np.random.uniform(0, 2 * np.pi)
#             rotation_matrix = np.array([
#                 [np.cos(theta), -np.sin(theta), 0],
#                 [np.sin(theta),  np.cos(theta), 0],
#                 [0,              0,             1]
#             ])
#             data = data @ rotation_matrix.T

#         # Normalize per point cloud
#         mean = np.mean(data, axis=0)
#         data = data - mean  # Center the point cloud

#         data = torch.tensor(data, dtype=torch.float32)
#         data = data.T  # Transpose to shape (3, num_points)
#         return data, data  # Return (input, target)

# # Placeholder for Graph Generation Utility
# def generate_from_conf(graph_conf):
#     """
#     Generates a communication graph based on the configuration.

#     Args:
#         graph_conf: Dictionary containing graph configuration.

#     Returns:
#         N: Number of nodes.
#         graph: NetworkX graph.
#     """
#     graph_type = graph_conf.get("type", "fully_connected")
#     num_nodes = graph_conf.get("num_nodes", 10)
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
#         graph = nx.erdos_renyi_graph(num_nodes, p, seed=None, directed=False)
#         attempts = 0
#         while not nx.is_connected(graph) and attempts < gen_attempts:
#             graph = nx.erdos_renyi_graph(num_nodes, p, seed=None, directed=False)
#             attempts += 1
#         if not nx.is_connected(graph):
#             raise ValueError("Failed to generate a connected Erdos-Renyi graph.")
#     else:
#         raise ValueError(f"Unknown graph type: {graph_type}")

#     return num_nodes, graph

# # Define DDLProblem Class
# class DDLProblem:
#     def __init__(self, models, N, conf, train_subsets, val_set):
#         self.models = models
#         self.N = N
#         self.conf = conf
#         # Calculate the total number of parameters in the first model
#         self.n = torch.numel(torch.nn.utils.parameters_to_vector(self.models[0].parameters()))
#         # Initialize communication graph
#         self.graph = generate_from_conf(conf["graph"])[1]
#         # Initialize data loaders for each node with drop_last=False to retain all data
#         self.train_loaders = [
#             DataLoader(
#                 train_subsets[i],
#                 batch_size=conf["train_batch_size"],
#                 shuffle=True,
#                 drop_last=False  # Ensures all data is processed
#             )
#             for i in range(N)
#         ]
#         self.val_loader = DataLoader(
#             val_set,
#             batch_size=conf["val_batch_size"],
#             shuffle=False,
#             drop_last=False  # Ensures all validation data is processed
#         )
#         # Assign device
#         self.device = torch.device(conf["device"])
#         # Initialize iterators for each train loader
#         self.train_iters = [iter(loader) for loader in self.train_loaders]

#     def local_batch_loss(self, i):
#         """
#         Compute the local batch loss for node i using Mean Squared Error loss.
#         """
#         model = self.models[i].to(self.device)
#         model.train()
#         try:
#             data, target = next(self.train_iters[i])
#         except StopIteration:
#             # Restart the loader if the iterator is exhausted
#             self.train_iters[i] = iter(self.train_loaders[i])
#             data, target = next(self.train_iters[i])
#         data, target = data.to(self.device), target.to(self.device)
#         output = model(data)
#         # Use Mean Squared Error loss
#         loss_mse = F.mse_loss(output, target)
#         return loss_mse

#     def evaluate_metrics(self, at_end=False, iteration=0):
#         """
#         Evaluate and return metrics such as validation loss using Mean Squared Error loss.
#         """
#         metrics = {}
#         for i, model in enumerate(self.models):
#             model.eval()
#             total_loss = 0.0
#             with torch.no_grad():
#                 for data, target in self.val_loader:
#                     data, target = data.to(self.device), target.to(self.device)
#                     output = model(data)
#                     loss_mse = F.mse_loss(output, target)
#                     total_loss += loss_mse.item()
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

# # Define DiNNO Optimizer Class
# class DiNNO:
#     def __init__(self, ddl_problem, device, conf):
#         self.pr = ddl_problem
#         self.conf = conf

#         # Initialize dual variables
#         self.duals = {
#             i: torch.zeros((self.pr.n), device=device)
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
#             self.primal_lr = np.linspace(
#                 self.conf["primal_lr_start"],
#                 self.conf["primal_lr_finish"],
#                 self.conf["outer_iterations"],
#             )
#         elif self.conf["lr_decay_type"] == "log":
#             self.primal_lr = np.logspace(
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
#                         self.pr.models[i].parameters(), self.primal_lr[0]
#                     )
#                 elif self.conf["primal_optimizer"] == "sgd":
#                     self.opts[i] = torch.optim.SGD(
#                         self.pr.models[i].parameters(), self.primal_lr[0]
#                     )
#                 elif self.conf["primal_optimizer"] == "adamw":
#                     self.opts[i] = torch.optim.AdamW(
#                         self.pr.models[i].parameters(), self.primal_lr[0]
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
#             # Update learning rate
#             for param_group in opt.param_groups:
#                 param_group['lr'] = self.primal_lr[k]
#         else:
#             if self.conf["primal_optimizer"] == "adam":
#                 opt = torch.optim.Adam(
#                     self.pr.models[i].parameters(), self.primal_lr[k]
#                 )
#             elif self.conf["primal_optimizer"] == "sgd":
#                 opt = torch.optim.SGD(
#                     self.pr.models[i].parameters(), self.primal_lr[k]
#                 )
#             elif self.conf["primal_optimizer"] == "adamw":
#                 opt = torch.optim.AdamW(
#                     self.pr.models[i].parameters(), self.primal_lr[k]
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
#                 torch.square(th - th_reg)
#             )

#             loss = pred_loss + torch.dot(th, self.duals[i]) + self.rho * reg
#             loss.backward()

#             # Gradient clipping to prevent exploding gradients
#             torch.nn.utils.clip_grad_norm_(self.pr.models[i].parameters(), max_norm=1.0)

#             opt.step()

#         return

#     def synchronize_weights(self):
#         """
#         Synchronize model weights with neighboring nodes by averaging.
#         """
#         for i in range(self.pr.N):
#             neighbors = list(self.pr.graph.neighbors(i))
#             if neighbors:
#                 # Collect weights from neighbors
#                 neighbor_weights = []
#                 for neighbor in neighbors:
#                     neighbor_weights.append(torch.nn.utils.parameters_to_vector(
#                         self.pr.models[neighbor].parameters()
#                     ))
#                 # Average the weights
#                 avg_weights = torch.mean(torch.stack(neighbor_weights), dim=0)
#                 # Get current model weights
#                 current_weights = torch.nn.utils.parameters_to_vector(
#                     self.pr.models[i].parameters()
#                 )
#                 # Update local model weights by averaging with neighbor's weights
#                 new_weights = (current_weights + avg_weights) / 2.0
#                 torch.nn.utils.vector_to_parameters(new_weights, self.pr.models[i].parameters())

#                 # Debugging: Print a confirmation
#                 if self.conf["verbose"]:
#                     print(f"Node {i} weights synchronized with neighbors.")

#     def train(self, profiler=None):
#         eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
#         oits = self.conf["outer_iterations"]
#         for k in range(oits):
#             if k % eval_every == 0 or k == oits - 1:
#                 # Evaluate metrics and append to self.metrics
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
#                 else:
#                     thj_mean = ths[i]

#                 self.duals[i] += self.rho * (ths[i] - thj_mean)
#                 th_reg = thj_mean
#                 self.primal_update(i, th_reg, k)

#             # Synchronize weights after each outer iteration
#             self.synchronize_weights()

#             if profiler is not None:
#                 profiler.step()

#         return

# # Define Reconstruction Function
# def reconstruct_maps(ddl_problem, device, output_dir):
#     """
#     Reconstruct and save the point clouds from each node.

#     Args:
#         ddl_problem: Instance of DDLProblem containing models and data loaders.
#         device: Torch device.
#         output_dir: Directory to save the reconstructed maps.

#     Returns:
#         reconstructed_point_clouds: List of numpy arrays for each node.
#     """
#     reconstructed_point_clouds = []
#     for i in range(ddl_problem.N):
#         model = ddl_problem.models[i].to(device)
#         model.eval()
#         all_reconstructions = []
#         with torch.no_grad():
#             for data, _ in ddl_problem.train_loaders[i]:
#                 data = data.to(device)
#                 output = model(data)
#                 output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
#                 output = output.cpu().numpy().reshape(-1, 3)
#                 all_reconstructions.append(output)
#         reconstructed_points = np.concatenate(all_reconstructions, axis=0)
#         reconstructed_point_clouds.append(reconstructed_points)

#         # Save the reconstructed points
#         np.save(os.path.join(output_dir, f"reconstructed_maps/reconstructed_map_node_{i}.npy"), reconstructed_points)
#         print(f"Reconstructed map for node {i} saved.")

#         # Visualize and save the plot
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2], s=0.5)
#         ax.set_title(f"Reconstructed Map Node {i}")
#         plt.savefig(os.path.join(output_dir, f"reconstructed_maps/reconstructed_map_node_{i}.png"))
#         plt.close(fig)
#         print(f"Reconstructed map for node {i} visualized and saved.")

#     return reconstructed_point_clouds

# # Function to save and visualize original point clouds
# def save_and_visualize_original_maps(node_point_clouds, output_dir):
#     """
#     Save and visualize the original point clouds from each node.

#     Args:
#         node_point_clouds: List of lists containing point clouds for each node.
#         output_dir: Directory to save the original maps.

#     Returns:
#         original_point_clouds: List of numpy arrays for each node.
#     """
#     original_point_clouds = []
#     for i, point_clouds in enumerate(node_point_clouds):
#         # Combine point clouds for the node
#         combined_points = np.concatenate(point_clouds, axis=0)
#         original_point_clouds.append(combined_points)

#         # Save the original points
#         np.save(os.path.join(output_dir, f"original_maps/original_map_node_{i}.npy"), combined_points)
#         print(f"Original map for node {i} saved.")

#         # Visualize and save the plot
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(combined_points[:, 0], combined_points[:, 1], combined_points[:, 2], s=0.5)
#         ax.set_title(f"Original Map Node {i}")
#         plt.savefig(os.path.join(output_dir, f"original_maps/original_map_node_{i}.png"))
#         plt.close(fig)
#         print(f"Original map for node {i} visualized and saved.")

#     return original_point_clouds

# # Main Training Function
# def train_dinno(ddl_problem, device, conf):
#     # Define the DiNNO optimizer
#     optimizer = DiNNO(ddl_problem, device, conf)

#     # Start training
#     optimizer.train()

#     return optimizer.metrics

# # Function to read KITTI Velodyne .bin file
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

# # Main Execution Block
# if __name__ == "__main__":
#     # Configuration
#     conf = {
#         "output_metadir": "E:/Documents/Masters_Courses/ESE 5460/final_project_env/DL_Project/output",
#         "name": "3d_map_DiNNO",
#         "epochs": 100,  # Corresponds to 'outer_iterations' in DiNNO
#         "verbose": True,
#         "graph": {
#             "type": "cycle",  # Options: "fully_connected", "cycle", "ring", "star", "erdos_renyi"
#             "num_nodes": 2,  # Adjusted number of nodes
#             "p": 0.3,
#             "gen_attempts": 100
#         },
#         "train_batch_size": 8,
#         "val_batch_size": 8,
#         "data_dir": "E:/Documents/Masters_Courses/ESE 5460/final_project_env/DL_Project/2011_09_28_drive_0035_sync/2011_09_28/2011_09_28_drive_0035_sync/velodyne_points/data",
#         "model": {
#             "in_channels": 3,  # Updated for PointNet
#             "out_channels": 3,
#             "init_features": 3,
#             "kernel_size": 1,
#             "linear_width": 64
#         },
#         "loss": "MSE",  # Changed to Mean Squared Error
#         "use_cuda": torch.cuda.is_available(),
#         "individual_training": {
#             "train_solo": False,
#             "optimizer": "adam",
#             "lr": 0.0005,  # Adjusted learning rate
#             "verbose": True
#         },
#         # DiNNO Specific Hyperparameters
#         "rho_init": 0.05,               # Adjusted from 0.1
#         "rho_scaling": 1.05,            # Adjusted from 1.1
#         "lr_decay_type": "linear",      # Changed from "constant" to "linear"
#         "primal_lr_start": 0.0005,      # Adjusted from 0.001
#         "primal_lr_finish": 0.00005,    # Adjusted from 0.0001
#         "outer_iterations": 100,        # Number of outer iterations
#         "primal_iterations": 20,        # Increased from 10
#         "persistant_primal_opt": True,  # Use persistent primal optimizers
#         "primal_optimizer": "adam",     # Type of primal optimizer: 'adam', 'sgd', 'adamw'
#         "metrics_config": {             # Metrics configuration (if used)
#             "evaluate_frequency": 1     # Evaluate metrics every iteration
#         },
#         "device": DEVICE.type,           # 'cuda' or 'cpu'
#         "num_points": 1024               # Number of points per point cloud
#     }

#     # Create output directory
#     if not os.path.exists(conf["output_metadir"]):
#         os.makedirs(conf["output_metadir"], exist_ok=True)

#     # Create experiment output directory
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_dir = os.path.join(conf["output_metadir"], f"{conf['name']}_{timestamp}")
#     os.makedirs(output_dir, exist_ok=True)

#     # Create subdirectories for original and reconstructed maps
#     os.makedirs(os.path.join(output_dir, "original_maps"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "reconstructed_maps"), exist_ok=True)

#     # Load point cloud data from KITTI .bin files
#     all_point_clouds = []
#     for file_name in sorted(os.listdir(conf["data_dir"])):
#         if file_name.endswith('.bin'):
#             file_path = os.path.join(conf["data_dir"], file_name)
#             points = read_kitti_bin(file_path)
#             all_point_clouds.append(points)

#     print(f"Total point clouds loaded: {len(all_point_clouds)}")

#     # Adjust the number of nodes
#     num_nodes = conf["graph"]["num_nodes"]

#     # Split point clouds among nodes
#     node_point_clouds = [[] for _ in range(num_nodes)]
#     for idx, point_cloud in enumerate(all_point_clouds):
#         node_idx = idx % num_nodes  # Distribute in a round-robin fashion
#         node_point_clouds[node_idx].append(point_cloud)

#     # Verify the distribution
#     for i in range(num_nodes):
#         print(f"Node {i} has {len(node_point_clouds[i])} point clouds.")

#     # Save and visualize original maps
#     print("Saving and visualizing original maps from each node...")
#     original_point_clouds = save_and_visualize_original_maps(node_point_clouds, output_dir)
#     print("Original maps saved and visualized.")

#     # Create validation set (use 10% of each node's data)
#     val_point_clouds = []
#     for i in range(num_nodes):
#         num_val_samples = max(1, len(node_point_clouds[i]) // 10)
#         val_point_clouds.extend(node_point_clouds[i][:num_val_samples])
#         node_point_clouds[i] = node_point_clouds[i][num_val_samples:]  # Remove validation data from training data

#     val_set = PointCloudDataset(
#         val_point_clouds,
#         num_points=conf["num_points"],
#         augment=False
#     )
#     print(f"Validation set size: {len(val_set)}")

#     # Create training subsets for each node
#     train_subsets = []
#     for i in range(num_nodes):
#         point_cloud_list = node_point_clouds[i]
#         dataset = PointCloudDataset(
#             point_cloud_list,
#             num_points=conf["num_points"],
#             augment=True
#         )
#         train_subsets.append(dataset)
#         print(f"Node {i} has {len(dataset)} training samples.")

#     # Create base models for each node
#     models = [PointNetAutoencoder(num_points=conf["num_points"], num_groups=32).to(DEVICE) for _ in range(num_nodes)]
#     print(f"Created {num_nodes} PointNetAutoencoders.")

#     # Verify Model Dtypes
#     for idx, model in enumerate(models):
#         for name, param in model.named_parameters():
#             print(f"Model {idx}, Parameter {name}, dtype: {param.dtype}")

#     # Create DDLProblem instance
#     ddl_problem = DDLProblem(models=models, N=num_nodes, conf=conf, train_subsets=train_subsets, val_set=val_set)
#     print("DDLProblem instance created.")

#     # Train using DiNNO
#     if conf["individual_training"]["train_solo"]:
#         print("Performing individual training...")
#         # Implement individual training logic here if needed
#         raise NotImplementedError("Individual training not implemented.")
#     else:
#         try:
#             metrics = train_dinno(ddl_problem, DEVICE, conf)
#         except Exception as e:
#             print(f"An error occurred during training: {e}")
#             metrics = None

#         if metrics is not None:
#             # Save metrics and models
#             torch.save(metrics, os.path.join(output_dir, "dinno_metrics.pt"))
#             for idx, model in enumerate(ddl_problem.models):
#                 torch.save(model.state_dict(), os.path.join(output_dir, f"dinno_trained_model_{idx}.pth"))
#             print("Training complete. Metrics and models saved.")

#             # Reconstruct and save the maps from each node
#             print("Reconstructing maps from each node...")
#             reconstructed_point_clouds = reconstruct_maps(ddl_problem, DEVICE, output_dir)
#             print("Reconstruction complete.")

#             # Combine and visualize the global map
#             print("Combining and visualizing the global map...")
#             global_map_points = np.vstack(reconstructed_point_clouds)
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             ax.scatter(global_map_points[:, 0], global_map_points[:, 1], global_map_points[:, 2], s=0.5)
#             ax.set_title("Combined Global Map")
#             plt.savefig(os.path.join(output_dir, "reconstructed_maps/combined_global_map.png"))
#             plt.close(fig)
#             print("Combined global map visualized and saved.")

#             # Save combined global map points
#             np.save(os.path.join(output_dir, "reconstructed_maps/combined_global_map.npy"), global_map_points)

#             # Visualize and save the combined original map
#             print("Visualizing and saving the combined original map...")
#             combined_original_points = np.vstack(original_point_clouds)
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             ax.scatter(combined_original_points[:, 0], combined_original_points[:, 1], combined_original_points[:, 2], s=0.5)
#             ax.set_title("Combined Original Map")
#             plt.savefig(os.path.join(output_dir, "original_maps/combined_original_map.png"))
#             plt.close(fig)
#             print("Combined original map visualized and saved.")

#             # Save combined original map points
#             np.save(os.path.join(output_dir, "original_maps/combined_original_map.npy"), combined_original_points)

#     print("Script execution completed.")


# Import necessary libraries
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import math
from datetime import datetime
import matplotlib.pyplot as plt
import open3d as o3d
import traceback  # For exception handling

# Check for CUDA availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is set to {"GPU" if DEVICE.type == "cuda" else "CPU"}')

# Define PointNet-Based Autoencoder with GroupNorm
class PointNetAutoencoder(nn.Module):
    def __init__(self, num_points=1024, num_groups=32):
        super(PointNetAutoencoder, self).__init__()
        self.num_points = num_points
        self.num_groups = num_groups  # Number of groups for GroupNorm

        # Encoder
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.gn1 = nn.GroupNorm(num_groups=self.num_groups, num_channels=64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.gn2 = nn.GroupNorm(num_groups=self.num_groups, num_channels=128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.gn3 = nn.GroupNorm(num_groups=self.num_groups, num_channels=1024)
        self.fc1 = nn.Linear(1024, 512)
        self.gn4 = nn.GroupNorm(num_groups=self.num_groups, num_channels=512)
        self.fc2 = nn.Linear(512, 256)
        self.gn5 = nn.GroupNorm(num_groups=self.num_groups, num_channels=256)
        
        # Decoder
        self.fc3 = nn.Linear(256, 512)
        self.gn6 = nn.GroupNorm(num_groups=self.num_groups, num_channels=512)
        self.fc4 = nn.Linear(512, 3 * self.num_points)

    def forward(self, x):
        # Encoder
        x = F.relu(self.gn1(self.conv1(x)))  # [batch_size, 64, num_points]
        x = F.relu(self.gn2(self.conv2(x)))  # [batch_size, 128, num_points]
        x = F.relu(self.gn3(self.conv3(x)))  # [batch_size, 1024, num_points]
        x = torch.max(x, 2, keepdim=True)[0]  # [batch_size, 1024, 1]
        x = x.view(-1, 1024)  # [batch_size, 1024]
        x = F.relu(self.gn4(self.fc1(x)))  # [batch_size, 512]
        x = F.relu(self.gn5(self.fc2(x)))  # [batch_size, 256]
        
        # Decoder
        x = F.relu(self.gn6(self.fc3(x)))  # [batch_size, 512]
        x = self.fc4(x)  # [batch_size, 3 * self.num_points]
        x = x.view(-1, 3, self.num_points)  # [batch_size, 3, num_points]
        return x

# Define Custom Dataset for Point Clouds
class PointCloudDataset(Dataset):
    def __init__(self, point_cloud_list, num_points=1024, augment=False):
        """
        Args:
            point_cloud_list: List of numpy arrays, each of shape (N_i, 3).
            num_points: Number of points to sample from each point cloud.
            augment: Whether to apply data augmentation.
        """
        self.point_cloud_list = point_cloud_list
        self.augment = augment
        self.num_points = num_points

    def __len__(self):
        return len(self.point_cloud_list)

    def __getitem__(self, idx):
        point_cloud = self.point_cloud_list[idx]
        num_points_in_cloud = point_cloud.shape[0]

        if num_points_in_cloud >= self.num_points:
            # Randomly sample num_points
            indices = np.random.choice(num_points_in_cloud, self.num_points, replace=False)
            data = point_cloud[indices]
        else:
            # If fewer points, pad with random points
            pad_size = self.num_points - num_points_in_cloud
            pad = np.random.normal(0, 0.001, size=(pad_size, 3))
            data = np.concatenate([point_cloud, pad], axis=0)

        if self.augment:
            # Apply random rotation around Z-axis
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,              0,             1]
            ])
            data = data @ rotation_matrix.T

        # Normalize per point cloud
        mean = np.mean(data, axis=0)
        data = data - mean  # Center the point cloud

        data = torch.tensor(data, dtype=torch.float32)
        data = data.T  # Transpose to shape (3, num_points)
        return data, data  # Return (input, target)

# Graph Generation Utility
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
        graph = nx.cycle_graph(num_nodes)
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
        # Initialize data loaders for each node with drop_last=False to retain all data
        self.train_loaders = [
            DataLoader(
                train_subsets[i],
                batch_size=conf["train_batch_size"],
                shuffle=True,
                drop_last=False  # Ensures all data is processed
            )
            for i in range(N)
        ]
        self.val_loader = DataLoader(
            val_set,
            batch_size=conf["val_batch_size"],
            shuffle=False,
            drop_last=False  # Ensures all validation data is processed
        )
        # Assign device
        self.device = torch.device(conf["device"])
        # Initialize iterators for each train loader
        self.train_iters = [iter(loader) for loader in self.train_loaders]

    def local_batch_loss(self, i):
        """
        Compute the local batch loss for node i using Mean Squared Error loss.
        """
        model = self.models[i].to(self.device)
        model.train()
        try:
            data, target = next(self.train_iters[i])
        except StopIteration:
            # Restart the loader if the iterator is exhausted
            self.train_iters[i] = iter(self.train_loaders[i])
            data, target = next(self.train_iters[i])
        data, target = data.to(self.device), target.to(self.device)
        output = model(data)
        # Use Mean Squared Error loss
        loss_mse = F.mse_loss(output, target)
        return loss_mse

    def evaluate_metrics(self, at_end=False, iteration=0):
        """
        Evaluate and return metrics such as validation loss using Mean Squared Error loss.
        """
        metrics = {}
        for i, model in enumerate(self.models):
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss_mse = F.mse_loss(output, target)
                    total_loss += loss_mse.item()
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
            self.primal_lr = np.linspace(
                self.conf["primal_lr_start"],
                self.conf["primal_lr_finish"],
                self.conf["outer_iterations"],
            )
        elif self.conf["lr_decay_type"] == "log":
            self.primal_lr = np.logspace(
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

        # Early Stopping parameters
        self.best_loss = float('inf')
        self.patience = 20  # Number of iterations to wait before stopping
        self.counter = 0

    def primal_update(self, i, th_reg, k):
        if self.conf["persistant_primal_opt"]:
            opt = self.opts[i]
            # Update learning rate
            for param_group in opt.param_groups:
                param_group['lr'] = self.primal_lr[k]
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
                torch.square(th - th_reg)
            )

            loss = pred_loss + torch.dot(th, self.duals[i]) + self.rho * reg
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.pr.models[i].parameters(), max_norm=1.0)

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

                # Debugging: Print a confirmation
                if self.conf["verbose"]:
                    print(f"Node {i} weights synchronized with neighbors.")

    def train(self, profiler=None):
        eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
        oits = self.conf["outer_iterations"]
        for k in range(oits):
            if k % eval_every == 0 or k == oits - 1:
                # Evaluate metrics and append to self.metrics
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
                else:
                    thj_mean = ths[i]

                self.duals[i] += self.rho * (ths[i] - thj_mean)
                th_reg = thj_mean
                self.primal_update(i, th_reg, k)

            # Synchronize weights after each outer iteration
            self.synchronize_weights()

            if profiler is not None:
                profiler.step()

        return

# Function to reconstruct maps and visualize
def reconstruct_maps(ddl_problem, device, output_dir):
    """
    Reconstruct and display the point clouds from each node using Open3D and Matplotlib.
    Each node's data is plotted in a different color.
    """
    reconstructed_point_clouds = []
    colors = ['green', 'red', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'gray']
    for i in range(ddl_problem.N):
        model = ddl_problem.models[i].to(device)
        model.eval()
        all_reconstructions = []
        with torch.no_grad():
            for data, _ in ddl_problem.train_loaders[i]:
                data = data.to(device)
                output = model(data)
                output = output.permute(0, 2, 1)  # [batch_size, num_points, 3]
                output = output.cpu().numpy().reshape(-1, 3)
                all_reconstructions.append(output)
        reconstructed_points = np.concatenate(all_reconstructions, axis=0)
        reconstructed_point_clouds.append(reconstructed_points)

        # Save the reconstructed points
        np.save(os.path.join(output_dir, f"reconstructed_maps/reconstructed_map_node_{i}.npy"), reconstructed_points)
        print(f"Reconstructed map for node {i} saved.")

        # Visualize and save the plot using Matplotlib with different colors
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2],
                   s=0.5, color=colors[i % len(colors)], label=f'Node {i}')
        ax.set_title(f"Reconstructed Map Node {i}")
        ax.legend()
        plt.savefig(os.path.join(output_dir, f"reconstructed_maps/reconstructed_map_node_{i}.png"))
        plt.close(fig)
        print(f"Reconstructed map for node {i} visualized and saved with Matplotlib.")

    return reconstructed_point_clouds

# Function to save and visualize original point clouds using Matplotlib
def save_and_visualize_original_maps(node_point_clouds, output_dir):
    """
    Save and display the original point clouds from each node using Matplotlib.
    Each node's data is plotted in a different color.
    """
    original_point_clouds = []
    colors = ['green', 'red', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'gray']
    for i, point_clouds in enumerate(node_point_clouds):
        try:
            print(f"Processing node {i}, number of point clouds: {len(point_clouds)}")
            # Combine point clouds for the node
            combined_points = np.concatenate(point_clouds, axis=0)
            print(f"Node {i}, combined_points shape: {combined_points.shape}")

            if combined_points.size == 0:
                print(f"Node {i} has an empty point cloud.")
                continue  # Skip visualization for this node

            # Save the original points
            np.save(os.path.join(output_dir, f"original_maps/original_map_node_{i}.npy"), combined_points)
            print(f"Original map for node {i} saved.")

            # Visualize and save the plot using Matplotlib with different colors
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(combined_points[:, 0], combined_points[:, 1], combined_points[:, 2],
                       s=0.5, color=colors[i % len(colors)], label=f'Node {i}')
            ax.set_title(f"Original Map Node {i}")
            ax.legend()
            plt.savefig(os.path.join(output_dir, f"original_maps/original_map_node_{i}.png"))
            plt.close(fig)
            print(f"Original map for node {i} visualized and saved with Matplotlib.")

            original_point_clouds.append(combined_points)

        except Exception as e:
            print(f"An error occurred while processing node {i}: {e}")
            traceback.print_exc()

    return original_point_clouds

# Main Training Function
def train_dinno(ddl_problem, device, conf):
    # Define the DiNNO optimizer
    optimizer = DiNNO(ddl_problem, device, conf)

    # Start training
    optimizer.train()

    return optimizer.metrics

# Function to read KITTI Velodyne .bin file
def read_kitti_bin(file_path):
    """
    Read a KITTI Velodyne .bin file and extract 3D point cloud (x, y, z).
    """
    try:
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        return points[:, :3]  # Extract only x, y, z coordinates
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return np.empty((0, 3))

# Main Execution Block
if __name__ == "__main__":
    try:
        # Configuration
        conf = {
            "output_metadir": "E:/Documents/Masters_Courses/ESE 5460/final_project_env/DL_Project/",
            "name": "3d_map_DiNNO",
            "epochs": 100,  # Corresponds to 'outer_iterations' in DiNNO
            "verbose": True,
            "graph": {
                "type": "cycle",  # Options: "fully_connected", "cycle", "ring", "star", "erdos_renyi"
                "num_nodes": 2,  # Adjust the number of nodes as needed
                "p": 0.3,
                "gen_attempts": 100
            },
            "train_batch_size": 8,
            "val_batch_size": 8,
            "data_dir": "E:/Documents/Masters_Courses/ESE 5460/final_project_env/DL_Project/2011_09_28_drive_0035_sync/2011_09_28/2011_09_28_drive_0035_sync/velodyne_points/data",
            "model": {
                "in_channels": 3,  # Updated for PointNet
                "out_channels": 3,
                "init_features": 3,
                "kernel_size": 1,
                "linear_width": 64
            },
            "loss": "MSE",  # Using Mean Squared Error
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
            "device": DEVICE.type,           # 'cuda' or 'cpu'
            "num_points": 1024               # Number of points per point cloud
        }

        # Create output directory
        if not os.path.exists(conf["output_metadir"]):
            os.makedirs(conf["output_metadir"], exist_ok=True)

        # Create experiment output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(conf["output_metadir"], f"{conf['name']}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectories for original and reconstructed maps
        os.makedirs(os.path.join(output_dir, "original_maps"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reconstructed_maps"), exist_ok=True)

        # Load point cloud data from KITTI .bin files
        all_point_clouds = []
        for file_name in sorted(os.listdir(conf["data_dir"])):
            if file_name.endswith('.bin'):
                file_path = os.path.join(conf["data_dir"], file_name)
                points = read_kitti_bin(file_path)
                all_point_clouds.append((points, file_path))

        print(f"Total point clouds loaded: {len(all_point_clouds)}")

        # Compute mean positions of point clouds for spatial partitioning
        point_cloud_means = []
        for points, file_path in all_point_clouds:
            if points.shape[0] > 0:
                mean_pos = np.mean(points[:, :2], axis=0)  # Use x and y coordinates
                point_cloud_means.append((mean_pos, points))
            else:
                print(f"Empty point cloud in file: {file_path}")

        # Sort point clouds based on mean x-coordinate
        point_cloud_means.sort(key=lambda x: x[0][0])  # Sort by x-coordinate

        # Adjust the number of nodes
        num_nodes = conf["graph"]["num_nodes"]

        # Split point clouds into spatial regions
        node_point_clouds = [[] for _ in range(num_nodes)]
        num_point_clouds = len(point_cloud_means)
        region_size = num_point_clouds // num_nodes

        for idx in range(num_nodes):
            start_idx = idx * region_size
            end_idx = (idx + 1) * region_size if idx < num_nodes - 1 else num_point_clouds
            for mean_pos, points in point_cloud_means[start_idx:end_idx]:
                node_point_clouds[idx].append(points)

        # Verify the distribution
        for i in range(num_nodes):
            print(f"Node {i} has {len(node_point_clouds[i])} point clouds.")

        # Save and visualize original maps using Matplotlib
        print("Saving and displaying original maps from each node...")
        original_point_clouds = save_and_visualize_original_maps(node_point_clouds, output_dir)
        print("Original maps saved and displayed.")

        # Create validation set (use 10% of each node's data)
        val_point_clouds = []
        for i in range(num_nodes):
            num_val_samples = max(1, len(node_point_clouds[i]) // 10)
            val_point_clouds.extend(node_point_clouds[i][:num_val_samples])
            node_point_clouds[i] = node_point_clouds[i][num_val_samples:]  # Remove validation data from training data

        val_set = PointCloudDataset(
            val_point_clouds,
            num_points=conf["num_points"],
            augment=False
        )
        print(f"Validation set size: {len(val_set)}")

        # Create training subsets for each node
        train_subsets = []
        for i in range(num_nodes):
            point_cloud_list = node_point_clouds[i]
            dataset = PointCloudDataset(
                point_cloud_list,
                num_points=conf["num_points"],
                augment=True
            )
            train_subsets.append(dataset)
            print(f"Node {i} has {len(dataset)} training samples.")

        # Create base models for each node
        models = [PointNetAutoencoder(num_points=conf["num_points"], num_groups=32).to(DEVICE) for _ in range(num_nodes)]
        print(f"Created {num_nodes} PointNetAutoencoders.")

        # Verify Model Dtypes
        for idx, model in enumerate(models):
            for name, param in model.named_parameters():
                print(f"Model {idx}, Parameter {name}, dtype: {param.dtype}")

        # Create DDLProblem instance
        ddl_problem = DDLProblem(models=models, N=num_nodes, conf=conf, train_subsets=train_subsets, val_set=val_set)
        print("DDLProblem instance created.")

        # Train using DiNNO
        if conf["individual_training"]["train_solo"]:
            print("Performing individual training...")
            # Implement individual training logic here if needed
            raise NotImplementedError("Individual training not implemented.")
        else:
            try:
                metrics = train_dinno(ddl_problem, DEVICE, conf)
            except Exception as e:
                print(f"An error occurred during training: {e}")
                traceback.print_exc()
                metrics = None

            if metrics is not None:
                # Save metrics and models
                torch.save(metrics, os.path.join(output_dir, "dinno_metrics.pt"))
                for idx, model in enumerate(ddl_problem.models):
                    torch.save(model.state_dict(), os.path.join(output_dir, f"dinno_trained_model_{idx}.pth"))
                print("Training complete. Metrics and models saved.")

                # Reconstruct and display the maps from each node using Matplotlib
                print("Reconstructing and displaying maps from each node...")
                reconstructed_point_clouds = reconstruct_maps(ddl_problem, DEVICE, output_dir)
                print("Reconstruction complete.")

                # Combine and display the global map using Matplotlib
                print("Combining and displaying the global map...")
                global_map_points = np.vstack(reconstructed_point_clouds)
                global_map_colors = []

                # Assign colors to each node's points
                colors = ['green', 'red', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'gray']
                for i, points in enumerate(reconstructed_point_clouds):
                    color = colors[i % len(colors)]
                    global_map_colors.extend([color] * points.shape[0])

                # Visualize and save using Matplotlib
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for i, points in enumerate(reconstructed_point_clouds):
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               s=0.5, color=colors[i % len(colors)], label=f'Node {i}')
                ax.set_title("Combined Global Map")
                ax.legend()
                plt.savefig(os.path.join(output_dir, "reconstructed_maps/combined_global_map.png"))
                plt.close(fig)
                print("Combined global map visualized and saved with Matplotlib.")

                # Save combined global map points
                np.save(os.path.join(output_dir, "reconstructed_maps/combined_global_map.npy"), global_map_points)

                # Display the combined original map using Matplotlib
                print("Displaying the combined original map...")
                combined_original_points = np.vstack(original_point_clouds)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for i, points in enumerate(original_point_clouds):
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               s=0.5, color=colors[i % len(colors)], label=f'Node {i}')
                ax.set_title("Combined Original Map")
                ax.legend()
                plt.savefig(os.path.join(output_dir, "original_maps/combined_original_map.png"))
                plt.close(fig)
                print("Combined original map visualized and saved with Matplotlib.")

                # Save combined original map points
                np.save(os.path.join(output_dir, "original_maps/combined_original_map.npy"), combined_original_points)

        print("Script execution completed.")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
