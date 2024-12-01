import torch
import numpy as np
import os

def compare_node_weights(output_dir, num_nodes=10):
    """
    Compares weights between nodes to ensure synchronization.
    
    Args:
        output_dir (str): Directory containing dinno_trained_model_X.pth files.
        num_nodes (int): Number of nodes.
    """
    print(f"Comparing model weights across {num_nodes} nodes in directory: {output_dir}")
    weight_vectors = []
    for i in range(num_nodes):
        model_path = os.path.join(output_dir, f"dinno_trained_model_{i}.pth")
        if not os.path.exists(model_path):
            print(f"Model for node {i} not found: {model_path}")
            continue

        try:
            # Load model state dict
            state_dict = torch.load(model_path, map_location='cpu')
            print(f"Loaded model for Node {i} from {model_path}")

            # Extract parameters and flatten to a vector
            param_list = []
            for key in sorted(state_dict.keys()):
                param_list.append(state_dict[key].view(-1))
            weight_vector = torch.cat(param_list).numpy()
            weight_vectors.append(weight_vector)
            print(f"Node {i}: Weight vector length: {weight_vector.shape[0]}")
        except Exception as e:
            print(f"Error loading model for Node {i}: {e}")
            continue

    # Compute pairwise differences
    print("\nComputing pairwise weight differences:")
    for i in range(len(weight_vectors)):
        for j in range(i+1, len(weight_vectors)):
            diff = np.linalg.norm(weight_vectors[i] - weight_vectors[j])
            print(f"Weight difference between Node {i} and Node {j}: {diff:.6f}")

if __name__ == "__main__":
    output_directory = "./output/"  # Update if different
    num_nodes = 10
    compare_node_weights(output_directory, num_nodes)
