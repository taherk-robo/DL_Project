# import os
# import numpy as np
# import open3d as o3d
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader

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

# # Neural Network (Autoencoder) for Point Cloud Data
# class PointCloudAutoencoder(nn.Module):
#     def __init__(self):
#         super(PointCloudAutoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(3, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 3)  # Reconstruct x, y, z
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

# # Main code to load data, train the neural network, and visualize point cloud
# if __name__ == "__main__":
#     # Path to the folder containing .bin files
#     folder_path = "/home/taherk/nn_distributed_training/2011_09_28_drive_0035_sync/velodyne_points/data"  # Replace with your actual folder path

#     # Step 1: Read and combine all .bin files into a single dataset
#     all_points = []
#     for file_name in sorted(os.listdir(folder_path)):
#         if file_name.endswith('.bin'):
#             file_path = os.path.join(folder_path, file_name)
#             points = read_kitti_bin(file_path)
#             all_points.append(points)

#     # Concatenate all point clouds into a single array
#     all_points = np.concatenate(all_points, axis=0)

#     # Visualize the combined point cloud
#     print("Visualizing combined point cloud...")
#     visualize_point_cloud(all_points)

#     # Step 2: Load data into DataLoader
#     dataset = PointCloudDataset(all_points)
#     dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

#     # Step 3: Initialize Neural Network, Loss, and Optimizer
#     model = PointCloudAutoencoder()
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # Step 4: Training Loop
#     print("Starting training...")
#     for epoch in range(10):  # Adjust epochs as needed
#         total_loss = 0
#         for batch in dataloader:
#             optimizer.zero_grad()
#             reconstructed = model(batch)
#             loss = criterion(reconstructed, batch)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

#     # Step 5: Save the trained model
#     torch.save(model.state_dict(), "pointcloud_autoencoder.pth")
#     print("Model saved as 'pointcloud_autoencoder.pth'.")

#     # Step 6: Visualize reconstructed point cloud (optional)
#     print("Visualizing reconstructed point cloud...")
#     with torch.no_grad():
#         reconstructed_points = model(torch.tensor(all_points, dtype=torch.float32)).numpy()
#         visualize_point_cloud(reconstructed_points)

# # import os
# # import numpy as np
# # import open3d as o3d
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import Dataset, DataLoader

# # def read_kitti_bin(file_path):
# #     """
# #     Read a KITTI Velodyne .bin file and extract 3D point cloud (x, y, z).
# #     :param file_path: Path to the .bin file.
# #     :return: numpy array of shape (N, 3), where N is the number of points.
# #     """
# #     points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
# #     return points[:, :3]

# # def visualize_point_cloud(points):
# #     """
# #     Visualize a 3D point cloud using Open3D.
# #     :param points: numpy array of shape (N, 3).
# #     """
# #     pcd = o3d.geometry.PointCloud()
# #     pcd.points = o3d.utility.Vector3dVector(points)
# #     o3d.visualization.draw_geometries([pcd])

# # if __name__ == "__main__":
# #     folder_path = "/home/taherk/Downloads/2011_09_28_drive_0035_sync/2011_09_28/2011_09_28_drive_0035_sync/velodyne_points/data"  # Replace with your actual folder path
# #     print("Files in folder:", os.listdir(folder_path))


# #     # Step 1: Read and combine all .bin files into a single dataset
# #     all_points = []
# #     for file_name in sorted(os.listdir(folder_path)):
# #         if file_name.endswith('.bin'):
# #             file_path = os.path.join(folder_path, file_name)
# #             print(f"Processing file: {file_path}")
# #             try:
# #                 points = read_kitti_bin(file_path)
# #                 all_points.append(points)
# #             except Exception as e:
# #                 print(f"Failed to process {file_path}: {e}")

# #     if not all_points:
# #         print("No valid .bin files found in the folder. Please check the folder path and file extensions.")
# #         exit(1)

# #     # Concatenate all point clouds into a single array
# #     all_points = np.concatenate(all_points, axis=0)
# #     print(f"Combined point cloud shape: {all_points.shape}")

# #     # Visualize the combined point cloud
# #     visualize_point_cloud(all_points)


import os
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Function to read KITTI Velodyne .bin file
def read_kitti_bin(file_path):
    """
    Read a KITTI Velodyne .bin file and extract 3D point cloud (x, y, z).
    :param file_path: Path to the .bin file.
    :return: numpy array of shape (N, 3), where N is the number of points.
    """
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # Extract only x, y, z coordinates

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

# Neural Network (Autoencoder) for Point Cloud Data
class PointCloudAutoencoder(nn.Module):
    def __init__(self):
        super(PointCloudAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Reconstruct x, y, z
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Main code to load data, train the neural network, and visualize point cloud
if __name__ == "__main__":
    # Path to the folder containing .bin files
    folder_path = "/home/taherk/nn_distributed_training/2011_09_28_drive_0035_sync/velodyne_points/data"  # Replace with your actual folder path

    # Step 1: Read and combine all .bin files into a single dataset
    all_points = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.bin'):
            file_path = os.path.join(folder_path, file_name)
            points = read_kitti_bin(file_path)
            all_points.append(points)

    # Concatenate all point clouds into a single array
    all_points = np.concatenate(all_points, axis=0)

    # Visualize the combined point cloud
    print("Visualizing combined point cloud...")
    visualize_point_cloud(all_points)

    # Step 2: Load data into Dataset and split into training and validation sets
    dataset = PointCloudDataset(all_points)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Step 3: Load data into DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Step 4: Initialize Neural Network, Loss, and Optimizer
    model = PointCloudAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step 5: Training Loop
    print("Starting training...")
    for epoch in range(10):  # Adjust epochs as needed
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")

        # Step 6: Validation Loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")

    # Step 7: Save the trained model
    torch.save(model.state_dict(), "pointcloud_autoencoder.pth")
    print("Model saved as 'pointcloud_autoencoder.pth'.")

    # Step 8: Visualize reconstructed point cloud (optional)
    print("Visualizing reconstructed point cloud...")
    with torch.no_grad():
        reconstructed_points = model(torch.tensor(all_points, dtype=torch.float32)).numpy()
        visualize_point_cloud(reconstructed_points)
