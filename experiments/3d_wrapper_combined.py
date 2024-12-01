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
# import open3d as o3d
# import traceback  # For exception handling

# # Check for CUDA availability
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Device is set to {"GPU" if DEVICE.type == "cuda" else "CPU"}')

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
#         x = self.fc4(x)  # [batch_size, 3 * num_points]
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

# # Graph Generation Utility
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
#             self.primal_lr = self.conf["primal_lr_start"] * np.ones(
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

# # Function to reconstruct maps and visualize
# def reconstruct_maps(ddl_problem, device, output_dir):
#     """
#     Reconstruct and display the point clouds from each node using Matplotlib.
#     Each node's data is plotted in a different color.
#     """
#     reconstructed_point_clouds = []
#     colors = ['green', 'red', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'gray']
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
#         node_recon_dir = os.path.join(output_dir, "reconstructed_maps")
#         os.makedirs(node_recon_dir, exist_ok=True)
#         np.save(os.path.join(node_recon_dir, f"reconstructed_map_node_{i}.npy"), reconstructed_points)
#         print(f"Reconstructed map for node {i} saved.")

#         # Visualize and save the plot using Matplotlib with different colors
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2],
#                    s=0.5, color=colors[i % len(colors)], label=f'Node {i}')
#         ax.set_title(f"Reconstructed Map Node {i}")
#         ax.legend()
#         plt.savefig(os.path.join(node_recon_dir, f"reconstructed_map_node_{i}.png"))
#         plt.close(fig)
#         print(f"Reconstructed map for node {i} visualized and saved with Matplotlib.")

#     return reconstructed_point_clouds

# # Function to save and visualize original point clouds using Matplotlib
# def save_and_visualize_original_maps(regions, output_dir):
#     """
#     Save and display the original point clouds from each spatial region using Matplotlib.
#     Each region's data is plotted in a different color.
#     """
#     colors = ['green', 'red', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'gray']
#     for i, region in enumerate(regions):
#         try:
#             print(f"Processing region {i}, number of points: {region.shape[0]}")

#             if region.size == 0:
#                 print(f"Region {i} has no points.")
#                 continue  # Skip visualization for this region

#             # Save the original points
#             node_orig_dir = os.path.join(output_dir, "original_maps")
#             os.makedirs(node_orig_dir, exist_ok=True)
#             np.save(os.path.join(node_orig_dir, f"original_map_region_{i}.npy"), region)
#             print(f"Original map for region {i} saved.")

#             # Visualize and save the plot using Matplotlib with different colors
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             ax.scatter(region[:, 0], region[:, 1], region[:, 2],
#                        s=0.5, color=colors[i % len(colors)], label=f'Region {i}')
#             ax.set_title(f"Original Map Region {i}")
#             ax.legend()
#             plt.savefig(os.path.join(node_orig_dir, f"original_map_region_{i}.png"))
#             plt.close(fig)
#             print(f"Original map for region {i} visualized and saved with Matplotlib.")

#         except Exception as e:
#             print(f"An error occurred while processing region {i}: {e}")
#             traceback.print_exc()

#     return

# # Function to read KITTI Velodyne .bin file
# def read_kitti_bin(file_path):
#     """
#     Read a KITTI Velodyne .bin file and extract 3D point cloud (x, y, z).
#     """
#     try:
#         points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
#         return points[:, :3]  # Extract only x, y, z coordinates
#     except Exception as e:
#         print(f"Error reading file {file_path}: {e}")
#         return np.empty((0, 3))

# # Main Training Function
# def train_dinno(ddl_problem, device, conf):
#     # Define the DiNNO optimizer
#     optimizer = DiNNO(ddl_problem, device, conf)

#     # Start training
#     optimizer.train()

#     return optimizer.metrics

# # Function to perform spatial splitting
# def spatial_split(point_cloud, num_regions, overlap_ratio=0.1, axis=0):
#     """
#     Splits the entire point cloud into spatial regions with overlapping areas.

#     Args:
#         point_cloud (numpy.ndarray): Array of shape (N, 3).
#         num_regions (int): Number of regions (nodes).
#         overlap_ratio (float): Fraction of overlap between adjacent regions.
#         axis (int): Axis along which to split (0=X, 1=Y, 2=Z).

#     Returns:
#         List of numpy.ndarray: List containing point clouds for each region.
#     """
#     sorted_indices = np.argsort(point_cloud[:, axis])
#     sorted_points = point_cloud[sorted_indices]

#     # Calculate the total range and region size
#     total_range = sorted_points[:, axis].max() - sorted_points[:, axis].min()
#     region_size = total_range / num_regions
#     overlap_size = region_size * overlap_ratio

#     regions = []
#     for i in range(num_regions):
#         start = sorted_points[:, axis].min() + i * region_size - (overlap_size if i > 0 else 0)
#         end = start + region_size + (overlap_size if i < num_regions - 1 else 0)
#         region_mask = (sorted_points[:, axis] >= start) & (sorted_points[:, axis] < end)
#         region_points = sorted_points[region_mask]
#         regions.append(region_points)
#         print(f"Region {i}: {region_points.shape[0]} points, {['X','Y','Z'][axis]} between {start:.3f} and {end:.3f}")
    
#     return regions

# # Function to visualize spatial regions
# def visualize_regions(regions, axis=0, output_dir=None):
#     """
#     Visualizes the spatial regions along a specified axis.

#     Args:
#         regions (list of numpy.ndarray): List of point clouds per region.
#         axis (int): Axis along which to visualize (0=X, 1=Y, 2=Z).
#         output_dir (str): Directory to save the visualization. If None, display on screen.
#     """
#     plt.figure(figsize=(10, 8))
#     colors = plt.cm.get_cmap('tab10', len(regions))
    
#     for i, region in enumerate(regions):
#         if region.size == 0:
#             continue  # Skip empty regions
#         plt.scatter(region[:, axis], region[:, (axis + 1) % 3], s=1, color=colors(i), label=f'Region {i}')
    
#     plt.xlabel(['X', 'Y', 'Z'][axis])
#     plt.ylabel(['Y', 'Z', 'X'][(axis + 1) % 3])
#     plt.legend()
#     plt.title(f'Spatial Regions Split along {["X", "Y", "Z"][axis]}-axis')
#     if output_dir:
#         plt.savefig(os.path.join(output_dir, f"spatial_regions_axis_{['X', 'Y', 'Z'][axis]}.png"))
#         plt.close()
#         print(f"Spatial regions visualization saved to {output_dir}.")
#     else:
#         plt.show()

# # Main Execution Block
# if __name__ == "__main__":
#     try:
#         # Configuration
#         conf = {
#             "output_metadir": "E:/Documents/Masters_Courses/ESE 5460/final_project_env/DL_Project/output",
#             "name": "3d_map_DiNNO",
#             "epochs": 100,  # Corresponds to 'outer_iterations' in DiNNO
#             "verbose": True,
#             "graph": {
#                 "type": "cycle",  # Options: "fully_connected", "cycle", "ring", "star", "erdos_renyi"
#                 "num_nodes": 2,  # Adjust the number of nodes as needed
#                 "p": 0.3,
#                 "gen_attempts": 100
#             },
#             "train_batch_size": 8,
#             "val_batch_size": 8,
#             "data_dir": "E:/Documents/Masters_Courses/ESE 5460/final_project_env/DL_Project/2011_09_28_drive_0035_sync/2011_09_28/2011_09_28_drive_0035_sync/velodyne_points/data",
#             "model": {
#                 "in_channels": 3,  # Updated for PointNet
#                 "out_channels": 3,
#                 "init_features": 3,
#                 "kernel_size": 1,
#                 "linear_width": 64
#             },
#             "loss": "MSE",  # Using Mean Squared Error
#             "use_cuda": torch.cuda.is_available(),
#             "individual_training": {
#                 "train_solo": False,
#                 "optimizer": "adam",
#                 "lr": 0.0005,  # Adjusted learning rate
#                 "verbose": True
#             },
#             # DiNNO Specific Hyperparameters
#             "rho_init": 0.05,               # Adjusted from 0.1
#             "rho_scaling": 1.05,            # Adjusted from 1.1
#             "lr_decay_type": "linear",      # Changed from "constant" to "linear"
#             "primal_lr_start": 0.0005,      # Adjusted from 0.001
#             "primal_lr_finish": 0.00005,    # Adjusted from 0.0001
#             "outer_iterations": 100,        # Number of outer iterations
#             "primal_iterations": 20,        # Increased from 10
#             "persistant_primal_opt": True,  # Use persistent primal optimizers
#             "primal_optimizer": "adam",     # Type of primal optimizer: 'adam', 'sgd', 'adamw'
#             "metrics_config": {             # Metrics configuration (if used)
#                 "evaluate_frequency": 1     # Evaluate metrics every iteration
#             },
#             "device": DEVICE.type,           # 'cuda' or 'cpu'
#             "num_points": 1024               # Number of points per point cloud
#         }

#         # Create output directory
#         if not os.path.exists(conf["output_metadir"]):
#             os.makedirs(conf["output_metadir"], exist_ok=True)

#         # Create experiment output directory with timestamp
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_dir = os.path.join(conf["output_metadir"], f"{conf['name']}_{timestamp}")
#         os.makedirs(output_dir, exist_ok=True)

#         # Create subdirectories for original and reconstructed maps
#         os.makedirs(os.path.join(output_dir, "original_maps"), exist_ok=True)
#         os.makedirs(os.path.join(output_dir, "reconstructed_maps"), exist_ok=True)

#         # Load point cloud data from KITTI .bin files
#         all_points = []
#         for file_name in sorted(os.listdir(conf["data_dir"])):
#             if file_name.endswith('.bin'):
#                 file_path = os.path.join(conf["data_dir"], file_name)
#                 points = read_kitti_bin(file_path)
#                 if points.shape[0] > 0:
#                     all_points.append(points)
#                 else:
#                     print(f"Empty point cloud in file: {file_path}")

#         if len(all_points) == 0:
#             raise ValueError("No valid point clouds found in the specified data directory.")

#         all_points = np.concatenate(all_points, axis=0)
#         print(f"Total points loaded: {all_points.shape[0]}")

#         # Perform spatial split on all points
#         num_regions = conf["graph"]["num_nodes"]
#         overlap_ratio = 0.1  # 10% overlap
#         axis = 0  # Split along X-axis
#         spatial_regions = spatial_split(all_points, num_regions, overlap_ratio=overlap_ratio, axis=axis)

#         # Visualize spatial regions
#         visualize_regions(spatial_regions, axis=axis, output_dir=output_dir)

#         # Assign regions to nodes
#         # Each node gets its corresponding region (with overlap)
#         # Points in overlapping regions are duplicated across nodes
#         node_point_clouds = spatial_regions  # Direct assignment

#         # Save and visualize original maps using Matplotlib
#         print("Saving and displaying original maps from each spatial region...")
#         save_and_visualize_original_maps(node_point_clouds, output_dir)
#         print("Original maps saved and visualized.")

#         # Create validation set
#         # For each region, take 10% of its points for validation
#         val_point_clouds = []
#         train_point_clouds = []
#         for i, region in enumerate(node_point_clouds):
#             num_val_points = max(1, int(0.1 * region.shape[0]))
#             val_points = region[:num_val_points]
#             train_points = region[num_val_points:]
#             val_point_clouds.append(val_points)
#             train_point_clouds.append(train_points)
#             print(f"Region {i}: {train_points.shape[0]} training points, {val_points.shape[0]} validation points.")

#         # Create PointCloudDataset instances
#         val_set = PointCloudDataset(
#             [points for points in val_point_clouds],
#             num_points=conf["num_points"],
#             augment=False
#         )
#         print(f"Validation set size: {len(val_set)}")

#         train_subsets = []
#         for i in range(num_regions):
#             dataset = PointCloudDataset(
#                 [train_point_clouds[i]],
#                 num_points=conf["num_points"],
#                 augment=True
#             )
#             train_subsets.append(dataset)
#             print(f"Node {i} has {len(dataset)} training samples.")

#         # Create base models for each node
#         models = [PointNetAutoencoder(num_points=conf["num_points"], num_groups=32).to(DEVICE) for _ in range(num_regions)]
#         print(f"Created {num_regions} PointNetAutoencoders.")

#         # Verify Model Dtypes
#         for idx, model in enumerate(models):
#             for name, param in model.named_parameters():
#                 print(f"Model {idx}, Parameter {name}, dtype: {param.dtype}")

#         # Create DDLProblem instance
#         ddl_problem = DDLProblem(models=models, N=num_regions, conf=conf, train_subsets=train_subsets, val_set=val_set)
#         print("DDLProblem instance created.")

#         # Train using DiNNO
#         if conf["individual_training"]["train_solo"]:
#             print("Performing individual training...")
#             # Implement individual training logic here if needed
#             raise NotImplementedError("Individual training not implemented.")
#         else:
#             try:
#                 metrics = train_dinno(ddl_problem, DEVICE, conf)
#             except Exception as e:
#                 print(f"An error occurred during training: {e}")
#                 traceback.print_exc()
#                 metrics = None

#             if metrics is not None:
#                 # Save metrics and models
#                 torch.save(metrics, os.path.join(output_dir, "dinno_metrics.pt"))
#                 for idx, model in enumerate(ddl_problem.models):
#                     torch.save(model.state_dict(), os.path.join(output_dir, f"dinno_trained_model_{idx}.pth"))
#                 print("Training complete. Metrics and models saved.")

#                 # Reconstruct and display the maps from each node using Matplotlib
#                 print("Reconstructing and displaying maps from each node...")
#                 reconstructed_point_clouds = reconstruct_maps(ddl_problem, DEVICE, output_dir)
#                 print("Reconstruction complete.")

#                 # Combine and display the global map using Matplotlib
#                 print("Combining and displaying the global map...")
#                 global_map_points = np.vstack(reconstructed_point_clouds)
#                 colors = ['green', 'red', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'gray']

#                 # Visualize and save using Matplotlib
#                 fig = plt.figure()
#                 ax = fig.add_subplot(111, projection='3d')
#                 for i, points in enumerate(reconstructed_point_clouds):
#                     ax.scatter(points[:, 0], points[:, 1], points[:, 2],
#                                s=0.5, color=colors[i % len(colors)], label=f'Node {i}')
#                 ax.set_title("Combined Global Map")
#                 ax.legend()
#                 plt.savefig(os.path.join(output_dir, "reconstructed_maps/combined_global_map.png"))
#                 plt.close(fig)
#                 print("Combined global map visualized and saved with Matplotlib.")

#                 # Save combined global map points
#                 np.save(os.path.join(output_dir, "reconstructed_maps/combined_global_map.npy"), global_map_points)

#                 # Display the combined original map using Matplotlib
#                 print("Displaying the combined original map...")
#                 combined_original_points = np.vstack(node_point_clouds)
#                 fig = plt.figure()
#                 ax = fig.add_subplot(111, projection='3d')
#                 for i, points in enumerate(node_point_clouds):
#                     ax.scatter(points[:, 0], points[:, 1], points[:, 2],
#                                s=0.5, color=colors[i % len(colors)], label=f'Region {i}')
#                 ax.set_title("Combined Original Map")
#                 ax.legend()
#                 plt.savefig(os.path.join(output_dir, "original_maps/combined_original_map.png"))
#                 plt.close(fig)
#                 print("Combined original map visualized and saved with Matplotlib.")

#                 # Save combined original map points
#                 np.save(os.path.join(output_dir, "original_maps/combined_original_map.npy"), combined_original_points)

#         print("Script execution completed.")

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         traceback.print_exc()


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
    def __init__(self, point_cloud, num_points=1024, num_samples=1000, augment=False):
        """
        Args:
            point_cloud: Numpy array of shape (N, 3).
            num_points: Number of points to sample from each point cloud.
            num_samples: Number of samples to generate per node.
            augment: Whether to apply data augmentation.
        """
        self.point_cloud = point_cloud
        self.num_points = num_points
        self.num_samples = num_samples
        self.augment = augment

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        num_points_in_cloud = self.point_cloud.shape[0]

        if num_points_in_cloud >= self.num_points:
            # Randomly sample num_points
            indices = np.random.choice(num_points_in_cloud, self.num_points, replace=False)
            data = self.point_cloud[indices]
        else:
            # If fewer points, pad with random points
            pad_size = self.num_points - num_points_in_cloud
            pad = np.random.normal(0, 0.001, size=(pad_size, 3))
            data = np.concatenate([self.point_cloud, pad], axis=0)

        if self.augment:
            # Apply random rotation around Z-axis
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,              0,             1]
            ])
            data = data @ rotation_matrix.T

        # Normalize per sample
        mean = np.mean(data, axis=0)
        data = data - mean  # Center the point cloud

        data = torch.tensor(data, dtype=torch.float32)
        data = data.T  # Transpose to shape (3, num_points)
        mean = torch.tensor(mean, dtype=torch.float32)
        return data, data, mean  # Return (input, target, mean)

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
        # Initialize data loaders for each node
        self.train_loaders = [
            DataLoader(
                train_subsets[i],
                batch_size=conf["train_batch_size"],
                shuffle=True,
                drop_last=False
            )
            for i in range(N)
        ]
        self.val_loader = DataLoader(
            val_set,
            batch_size=conf["val_batch_size"],
            shuffle=False,
            drop_last=False
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
            data, target, _ = next(self.train_iters[i])
        except StopIteration:
            # Restart the loader if the iterator is exhausted
            self.train_iters[i] = iter(self.train_loaders[i])
            data, target, _ = next(self.train_iters[i])
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
                for data, target, _ in self.val_loader:
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
            self.primal_lr = self.conf["primal_lr_start"] * np.ones(
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

        loss_values = []
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
            loss_values.append(loss.item())

        avg_loss = sum(loss_values) / len(loss_values)
        return avg_loss

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
        self.training_loss_per_node = {i: [] for i in range(self.pr.N)}
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
                avg_loss = self.primal_update(i, th_reg, k)
                self.training_loss_per_node[i].append(avg_loss)

            # Synchronize weights after each outer iteration
            self.synchronize_weights()

            if profiler is not None:
                profiler.step()

        return

# Function to reconstruct maps and visualize
def reconstruct_maps(ddl_problem, device, output_dir):
    """
    Reconstruct and display the point clouds from each node using Matplotlib.
    Each node's data is plotted in a different color.
    """
    reconstructed_point_clouds = []
    colors = ['green', 'red', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'gray']
    for i in range(ddl_problem.N):
        model = ddl_problem.models[i].to(device)
        model.eval()
        all_reconstructions = []
        with torch.no_grad():
            for data, _, mean in ddl_problem.train_loaders[i]:
                data = data.to(device)
                output = model(data)
                output = output.permute(0, 2, 1).cpu().numpy()  # [batch_size, num_points, 3]
                mean = mean.cpu().numpy()
                output += mean[:, np.newaxis, :]  # Add mean back
                output = output.reshape(-1, 3)
                all_reconstructions.append(output)
        reconstructed_points = np.concatenate(all_reconstructions, axis=0)
        reconstructed_point_clouds.append(reconstructed_points)

        # Save the reconstructed points
        node_recon_dir = os.path.join(output_dir, "reconstructed_maps")
        os.makedirs(node_recon_dir, exist_ok=True)
        np.save(os.path.join(node_recon_dir, f"reconstructed_map_node_{i}.npy"), reconstructed_points)
        print(f"Reconstructed map for node {i} saved.")

        # Visualize and save the plot using Matplotlib with different colors
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2],
                   s=0.5, color=colors[i % len(colors)], label=f'Node {i}')
        ax.set_title(f"Reconstructed Map Node {i}")
        ax.legend()
        plt.savefig(os.path.join(node_recon_dir, f"reconstructed_map_node_{i}.png"))
        plt.close(fig)
        print(f"Reconstructed map for node {i} visualized and saved with Matplotlib.")

    return reconstructed_point_clouds

# Function to save and visualize original point clouds using Matplotlib
def save_and_visualize_original_maps(regions, output_dir):
    """
    Save and display the original point clouds from each spatial region using Matplotlib.
    Each region's data is plotted in a different color.
    """
    colors = ['green', 'red', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'gray']
    for i, region in enumerate(regions):
        try:
            print(f"Processing region {i}, number of points: {region.shape[0]}")

            if region.size == 0:
                print(f"Region {i} has no points.")
                continue  # Skip visualization for this region

            # Save the original points
            node_orig_dir = os.path.join(output_dir, "original_maps")
            os.makedirs(node_orig_dir, exist_ok=True)
            np.save(os.path.join(node_orig_dir, f"original_map_region_{i}.npy"), region)
            print(f"Original map for region {i} saved.")

            # Visualize and save the plot using Matplotlib with different colors
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(region[:, 0], region[:, 1], region[:, 2],
                       s=0.5, color=colors[i % len(colors)], label=f'Region {i}')
            ax.set_title(f"Original Map Region {i}")
            ax.legend()
            plt.savefig(os.path.join(node_orig_dir, f"original_map_region_{i}.png"))
            plt.close(fig)
            print(f"Original map for region {i} visualized and saved with Matplotlib.")

        except Exception as e:
            print(f"An error occurred while processing region {i}: {e}")
            traceback.print_exc()

    return

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

# Main Training Function
def train_dinno(ddl_problem, device, conf):
    # Define the DiNNO optimizer
    optimizer = DiNNO(ddl_problem, device, conf)

    # Start training
    optimizer.train()

    return optimizer.metrics, optimizer.training_loss_per_node

# Function to perform spatial splitting
def spatial_split(point_cloud, num_regions, overlap_ratio=0.1, axis=0):
    """
    Splits the entire point cloud into spatial regions with overlapping areas.

    Args:
        point_cloud (numpy.ndarray): Array of shape (N, 3).
        num_regions (int): Number of regions (nodes).
        overlap_ratio (float): Fraction of overlap between adjacent regions.
        axis (int): Axis along which to split (0=X, 1=Y, 2=Z).

    Returns:
        List of numpy.ndarray: List containing point clouds for each region.
    """
    sorted_indices = np.argsort(point_cloud[:, axis])
    sorted_points = point_cloud[sorted_indices]

    # Calculate the total range and region size
    total_range = sorted_points[:, axis].max() - sorted_points[:, axis].min()
    region_size = total_range / num_regions
    overlap_size = region_size * overlap_ratio

    regions = []
    for i in range(num_regions):
        start = sorted_points[:, axis].min() + i * region_size - (overlap_size if i > 0 else 0)
        end = start + region_size + (overlap_size if i < num_regions - 1 else 0)
        region_mask = (sorted_points[:, axis] >= start) & (sorted_points[:, axis] < end)
        region_points = sorted_points[region_mask]
        regions.append(region_points)
        print(f"Region {i}: {region_points.shape[0]} points, {['X','Y','Z'][axis]} between {start:.3f} and {end:.3f}")
    
    return regions

# Function to visualize spatial regions
def visualize_regions(regions, axis=0, output_dir=None):
    """
    Visualizes the spatial regions along a specified axis.

    Args:
        regions (list of numpy.ndarray): List of point clouds per region.
        axis (int): Axis along which to visualize (0=X, 1=Y, 2=Z).
        output_dir (str): Directory to save the visualization. If None, display on screen.
    """
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10', len(regions))
    
    for i, region in enumerate(regions):
        if region.size == 0:
            continue  # Skip empty regions
        plt.scatter(region[:, axis], region[:, (axis + 1) % 3], s=1, color=colors(i), label=f'Region {i}')
    
    plt.xlabel(['X', 'Y', 'Z'][axis])
    plt.ylabel(['Y', 'Z', 'X'][(axis + 1) % 3])
    plt.legend()
    plt.title(f'Spatial Regions Split along {["X", "Y", "Z"][axis]}-axis')
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"spatial_regions_axis_{['X', 'Y', 'Z'][axis]}.png"))
        plt.close()
        print(f"Spatial regions visualization saved to {output_dir}.")
    else:
        plt.show()

# Main Execution Block
if __name__ == "__main__":
    try:
        # Configuration
        conf = {
            "output_metadir": "E:/Documents/Masters_Courses/ESE 5460/final_project_env/DL_Project/output",
            "name": "3d_map_DiNNO",
            "epochs": 100,  # Corresponds to 'outer_iterations' in DiNNO
            "verbose": True,
            "graph": {
                "type": "cycle",  # Options: "fully_connected", "cycle", "ring", "star", "erdos_renyi"
                "num_nodes": 2,  # Adjust the number of nodes as needed
                "p": 0.3,
                "gen_attempts": 100
            },
            "train_batch_size": 16,
            "val_batch_size": 16,
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
            "outer_iterations": 50,         # Number of outer iterations
            "primal_iterations": 10,        # Adjust as needed
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

        # Create experiment output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(conf["output_metadir"], f"{conf['name']}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectories for original and reconstructed maps
        os.makedirs(os.path.join(output_dir, "original_maps"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reconstructed_maps"), exist_ok=True)

        # Load point cloud data from KITTI .bin files
        all_points = []
        for file_name in sorted(os.listdir(conf["data_dir"])):
            if file_name.endswith('.bin'):
                file_path = os.path.join(conf["data_dir"], file_name)
                points = read_kitti_bin(file_path)
                if points.shape[0] > 0:
                    all_points.append(points)
                else:
                    print(f"Empty point cloud in file: {file_path}")

        if len(all_points) == 0:
            raise ValueError("No valid point clouds found in the specified data directory.")

        all_points = np.concatenate(all_points, axis=0)
        print(f"Total points loaded: {all_points.shape[0]}")

        # Perform spatial split on all points
        num_regions = conf["graph"]["num_nodes"]
        overlap_ratio = 0.1  # 10% overlap
        axis = 0  # Split along X-axis
        spatial_regions = spatial_split(all_points, num_regions, overlap_ratio=overlap_ratio, axis=axis)

        # Visualize spatial regions
        visualize_regions(spatial_regions, axis=axis, output_dir=output_dir)

        # Assign regions to nodes
        # Each node gets its corresponding region (with overlap)
        # Points in overlapping regions are duplicated across nodes
        node_point_clouds = spatial_regions  # Direct assignment

        # Save and visualize original maps using Matplotlib
        print("Saving and displaying original maps from each spatial region...")
        save_and_visualize_original_maps(node_point_clouds, output_dir)
        print("Original maps saved and visualized.")

        # Create validation set
        val_point_clouds = []
        train_point_clouds = []
        for i, region in enumerate(node_point_clouds):
            num_val_points = max(1, int(0.1 * region.shape[0]))
            val_points = region[:num_val_points]
            train_points = region[num_val_points:]
            val_point_clouds.append(val_points)
            train_point_clouds.append(train_points)
            print(f"Region {i}: {train_points.shape[0]} training points, {val_points.shape[0]} validation points.")

        # Create PointCloudDataset instances
        num_samples_per_node = 1000  # Adjust as needed
        train_subsets = []
        for i in range(num_regions):
            dataset = PointCloudDataset(
                train_point_clouds[i],
                num_points=conf["num_points"],
                num_samples=num_samples_per_node,
                augment=True
            )
            train_subsets.append(dataset)
            print(f"Node {i} has {len(dataset)} training samples.")

        # Combine validation point clouds
        val_point_cloud_combined = np.concatenate(val_point_clouds, axis=0)
        val_set = PointCloudDataset(
            val_point_cloud_combined,
            num_points=conf["num_points"],
            num_samples=500,  # Adjust as needed
            augment=False
        )
        print(f"Validation set size: {len(val_set)}")

        # Create base models for each node
        models = [PointNetAutoencoder(num_points=conf["num_points"], num_groups=32).to(DEVICE) for _ in range(num_regions)]
        print(f"Created {num_regions} PointNetAutoencoders.")

        # Verify Model Dtypes
        for idx, model in enumerate(models):
            for name, param in model.named_parameters():
                print(f"Model {idx}, Parameter {name}, dtype: {param.dtype}")

        # Create DDLProblem instance
        ddl_problem = DDLProblem(models=models, N=num_regions, conf=conf, train_subsets=train_subsets, val_set=val_set)
        print("DDLProblem instance created.")

        # Train using DiNNO
        if conf["individual_training"]["train_solo"]:
            print("Performing individual training...")
            # Implement individual training logic here if needed
            raise NotImplementedError("Individual training not implemented.")
        else:
            try:
                metrics, training_loss_per_node = train_dinno(ddl_problem, DEVICE, conf)
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

                # Plot the training loss per node
                for i in range(ddl_problem.N):
                    loss_values = training_loss_per_node[i]
                    plt.figure()
                    plt.plot(loss_values)
                    plt.title(f"Training Loss per Iteration for Node {i}")
                    plt.xlabel("Iteration")
                    plt.ylabel("Loss")
                    plt.savefig(os.path.join(output_dir, f"training_loss_node_{i}.png"))
                    plt.close()
                print("Training loss plots saved.")

                # Reconstruct and display the maps from each node using Matplotlib
                print("Reconstructing and displaying maps from each node...")
                reconstructed_point_clouds = reconstruct_maps(ddl_problem, DEVICE, output_dir)
                print("Reconstruction complete.")

                # Combine and display the global map using Matplotlib
                print("Combining and displaying the global map...")
                global_map_points = np.vstack(reconstructed_point_clouds)
                colors = ['green', 'red', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'gray']

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
                combined_original_points = np.vstack(node_point_clouds)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for i, points in enumerate(node_point_clouds):
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               s=0.5, color=colors[i % len(colors)], label=f'Region {i}')
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
