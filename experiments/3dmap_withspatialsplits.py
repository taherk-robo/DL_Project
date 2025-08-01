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
import matplotlib.pyplot as plt  # For visualization

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
        # self.conv4 = nn.Conv1d(1024, 2048, 1)  # Uncomment if increasing model complexity
        # self.bn4 = nn.BatchNorm1d(2048)
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
        # x = F.relu(self.bn4(self.conv4(x)))  # [batch_size, 2048, num_points]
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
            # If fewer points, pad with random points
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

# Function to visualize spatial regions
def visualize_regions(point_clouds, regions, axis=0):
    """
    Visualizes the spatial regions along a specified axis.
    
    Args:
        point_clouds (numpy.ndarray): Original point cloud data.
        regions (list of numpy.ndarray): List of point clouds per region.
        axis (int): Axis along which to visualize (0=X, 1=Y, 2=Z).
    """
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10', len(regions))
    
    for i, region in enumerate(regions):
        plt.scatter(region[:, axis], region[:, (axis + 1) % 3], s=1, color=colors(i), label=f'Region {i}')
    
    plt.xlabel(['X', 'Y', 'Z'][axis])
    plt.ylabel(['Y', 'Z', 'X'][(axis + 1) % 3])
    plt.legend()
    plt.title(f'Spatial Regions Split along {["X", "Y", "Z"][axis]}-axis')
    plt.show()

# Function to perform spatial splitting
def spatial_split(point_clouds, num_regions, overlap_ratio=0.1):
    """
    Splits point clouds into spatial regions with overlapping areas.
    
    Args:
        point_clouds (numpy.ndarray): Array of shape (N, 3).
        num_regions (int): Number of regions (robots).
        overlap_ratio (float): Fraction of overlap between adjacent regions.
        
    Returns:
        List of numpy.ndarray: List containing point clouds for each region.
    """
    # Determine the split axis (e.g., X-axis)
    # You can choose based on data distribution; here we choose X-axis
    axis = 0  # 0 for X, 1 for Y, 2 for Z
    sorted_indices = np.argsort(point_clouds[:, axis])
    sorted_points = point_clouds[sorted_indices]
    
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
        print(f"Region {i}: {region_points.shape[0]} points, X between {start} and {end}")
    
    return regions

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
        self.device = torch.device(conf["device"])
        # Initialize iterators for each train loader
        self.train_iters = [iter(loader) for loader in self.train_loaders]

    def local_batch_loss(self, i):
        """
        Compute the local batch loss for node i using the appropriate Chamfer Distance.
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

        # Early Stopping parameters
        self.best_loss = float('inf')
        self.patience = 20  # Increased patience due to distributed training complexity
        self.counter = 0

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
        "train_batch_size": 8,          # Reduced from 16
        "val_batch_size": 8,            # Reduced from 16
        "data_split_type": "spatial",   # Changed from "random" to "spatial"
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
            "lr": 0.0005,  # Reduced from 0.001
            "verbose": True
        },
        # DiNNO Specific Hyperparameters
        "rho_init": 0.05,               # Reduced from 0.1
        "rho_scaling": 1.05,            # Reduced from 1.1
        "lr_decay_type": "linear",      # Changed from "constant" to "linear"
        "primal_lr_start": 0.0005,      # Reduced from 0.001
        "primal_lr_finish": 0.00005,    # Reduced from 0.0001
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

    # Visualize Spatial Regions (optional)
    # You can uncomment the following lines to visualize the spatial split
    # Define the number of regions (robots)
    num_regions = conf["graph"]["num_nodes"]
    # Define the overlap ratio (e.g., 10% of the region size)
    overlap_ratio = 0.1
    # Perform spatial splitting
    spatial_regions = spatial_split(all_points, num_regions, overlap_ratio=overlap_ratio)
    # Visualize the regions along X-axis
    visualize_regions(all_points, spatial_regions, axis=0)

    # Create training subsets for each node based on spatial regions
    train_subsets = []
    for i in range(num_regions):
        region_points = spatial_regions[i]
        # Create a PointCloudDataset for each region
        dataset = PointCloudDataset(region_points, num_points=conf["num_points"], augment=True)
        train_subsets.append(dataset)

    print("Data split spatially among nodes with overlapping regions.")

    # Create validation set
    # Choose regions that are common overlaps or specific areas
    # For simplicity, using the first region's overlapping part as validation
    val_region_points = spatial_regions[0]
    val_set = PointCloudDataset(val_region_points, num_points=conf["num_points"], augment=False)
    print(f"Validation set size: {len(val_set)}")

    # Create base models for each node
    models = [PointNetAutoencoder(num_points=conf["num_points"]).to(DEVICE) for _ in range(num_regions)]
    print(f"Created {num_regions} PointNetAutoencoders.")

    # Verify Model Dtypes
    for idx, model in enumerate(models):
        for name, param in model.named_parameters():
            print(f"Model {idx}, Parameter {name}, dtype: {param.dtype}")

    # Create DDLProblem instance
    ddl_problem = DDLProblem(models=models, N=num_regions, conf=conf, train_subsets=train_subsets, val_set=val_set)
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
