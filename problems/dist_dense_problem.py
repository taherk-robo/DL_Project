import torch
import copy
import os
import numpy as np
from torch._C import device


class DistDensityProblem:
    def __init__(
        self,
        graph,
        base_model,
        base_loss,
        train_sets,
        val_set,
        device,
        conf,
    ):
        self.graph = graph
        self.base_loss = base_loss
        self.train_sets = train_sets
        self.val_set = val_set
        self.conf = conf

        self.N = graph.number_of_nodes()
        self.n = torch.nn.utils.parameters_to_vector(
            base_model.parameters()
        ).shape[0]

        # Copy the base model for each node
        self.models = {i: copy.deepcopy(base_model) for i in range(self.N)}

        # Create train loaders and iterators with the specified batch size
        self.train_loaders = {}
        self.train_iters = {}
        for i in range(self.N):
            self.train_loaders[i] = torch.utils.data.DataLoader(
                self.train_sets[i],
                batch_size=self.conf["train_batch_size"],
                shuffle=True,
                num_workers=4,
            )

            self.train_iters[i] = iter(self.train_loaders[i])

        self.val_loader = torch.utils.data.DataLoader(
            self.val_set, batch_size=self.conf["val_batch_size"]
        )

        # Initialize lists for metrics with names
        self.metrics = {met_name: [] for met_name in self.conf["metrics"]}
        self.epoch_tracker = torch.zeros(self.N)
        self.forward_cnt = 0

        if "mesh_grid_density" in self.metrics:
            X, Y = np.meshgrid(val_set.lidar.xs, val_set.lidar.ys)
            xlocs = X[::8, ::8].reshape(-1, 1)
            ylocs = Y[::8, ::8].reshape(-1, 1)
            mesh_poses = np.hstack((xlocs, ylocs))
            self.mesh_inputs = torch.Tensor(mesh_poses)

            # For reconstruction during visualization
            self.metrics["mesh_inputs"] = self.mesh_inputs

        # Device check for GPU
        self.device = device

        # send all of the models to the GPU
        for i in range(self.N):
            self.models[i] = self.models[i].to(self.device)

        self.mesh_inputs = self.mesh_inputs.to(self.device)

    def update_graph(self):
        """Placeholder because the graph is not dynamic in this problem."""
        pass

    def local_batch_loss(self, i):
        """Forward pass on a batch data for model at node i,
        and if it's node_id = 0 then increment a metric that
        counts the number of forward passes. Also increment an
        epoch tracker for each node when the iterator resets.

        Finally compute loss based on base_loss function and return.

        Note: if for whatever reason there isn't a node zero then
        this method's metric increment step will fail.

        Args:
            i (int): Node id

        Returns:
            (torch.Tensor): Loss of node i's model on a batch of
            local data.
        """
        try:
            locs, dens = next(self.train_iters[i])
        except StopIteration:
            self.epoch_tracker[i] += 1
            self.train_iters[i] = iter(self.train_loaders[i])
            locs, dens = next(self.train_iters[i])

        if i == 0:
            # Count the number of forward passes that have been performed
            # because this is symmetric across nodes we only have to do
            # this for node 0, and it will be consistent with all nodes.
            self.forward_cnt += self.conf["train_batch_size"]

        yh = self.models[i].forward(locs.to(self.device))

        return self.base_loss(torch.squeeze(yh), dens.to(self.device))

    def save_metrics(self, output_dir):
        """Save current metrics lists to a PT file."""
        file_name = self.conf["problem_name"] + "_results.pt"
        file_path = os.path.join(output_dir, file_name)
        torch.save(self.metrics, file_path)
        return

    def validate(self, i):
        """Compute the loss of a single node on the validation set.

        Args:
            i (int): Node id
        """
        val_loss = 0.0
        for batch in self.val_loader:
            with torch.no_grad():
                locs, dens = batch[0].to(self.device), batch[1].to(self.device)
                yh = self.models[i].forward(locs)
                val_loss += self.base_loss(
                    torch.squeeze(yh), dens
                ).data.detach()
        return val_loss

    def mesh_grid_density(self, i):
        """Computes the predicted density of the specified model on
        a mesh. This differs from the validation set because we should
        not count the loss of the model in regions where there is no
        training data (ie inside of walls).

        Args:
            i (int): Node id

        Returns:
            torch.Tensor: A vector of densities corresponding to the
            poses specified by the mesh that is created during initialization.
        """
        with torch.no_grad():
            mesh_density = self.models[i].forward(self.mesh_inputs)

        return mesh_density

    def evaluate_metrics(self):
        """Evaluate models, and then append values to the metric lists."""

        evalprint = "| "
        for met_name in self.conf["metrics"]:
            if met_name == "consensus_error":
                # The average distance from a single node to all
                # of the other nodes in the problem
                with torch.no_grad():
                    all_params = [
                        torch.nn.utils.parameters_to_vector(
                            self.models[i].parameters()
                        )
                        for i in range(self.N)
                    ]
                    # Stack all of the parameters into rows
                    th_stack = torch.stack(all_params)
                    # Normalize the stack
                    th_stack = torch.nn.functional.normalize(th_stack, dim=1)
                    # Compute row-wise distances
                    distances = torch.cdist(th_stack, th_stack)
                    davg = distances.sum(axis=1) / self.N
                # append metrics and generate print string
                self.metrics[met_name].append(distances)
                evalprint += "Consensus: {:.4f} - {:.4f} | ".format(
                    torch.amin(davg).item(), torch.amax(davg).item()
                )
            elif met_name == "validation_loss":
                # Average node loss on the validation dataset
                val_losses = [self.validate(i) for i in range(self.N)]
                val_losses = torch.tensor(val_losses)

                self.metrics[met_name].append(val_losses)
                evalprint += "Val Loss: {:.4f} - {:.4f} | ".format(
                    torch.amin(val_losses).item(),
                    torch.amax(val_losses).item(),
                )
            elif met_name == "mesh_grid_density":
                # Compute the mesh grid densities of each node
                # used for post run visualization so there is no
                # need to print anything.
                densities = [self.mesh_grid_density(i) for i in range(self.N)]
                self.metrics[met_name].append(torch.stack(densities))
            elif met_name == "forward_pass_count":
                # Number of forward passes performed by each node
                self.metrics[met_name].append(self.forward_cnt)
                evalprint += "Num Forward: {} | ".format(self.forward_cnt)
            elif met_name == "current_epoch":
                # Current epoch of each node (only different if the datasets at
                # each node are not the same size)
                self.metrics[met_name].append(
                    copy.deepcopy(self.epoch_tracker)
                )
                evalprint += "Ep Range: {} - {} | ".format(
                    int(torch.amin(self.epoch_tracker).item()),
                    int(torch.amax(self.epoch_tracker).item()),
                )
            else:
                raise NameError("Unknown metric.")

        print(evalprint)
        return
