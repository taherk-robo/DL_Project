# import torch
# import math


# class DiNNO:
#     def __init__(self, ddl_problem, device, conf):
#         self.pr = ddl_problem
#         self.conf = conf

#         self.duals = {
#             i: torch.zeros((self.pr.n), device=device)
#             for i in range(self.pr.N)
#         }

#         self.rho = self.conf["rho_init"]
#         self.rho_scaling = self.conf["rho_scaling"]
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
#                 math.log(self.conf["primal_lr_start"], 10),
#                 math.log(self.conf["primal_lr_finish"], 10),
#                 self.conf["outer_iterations"],
#             )
#         else:
#             raise NameError("Unknow primal learning rate decay type.")
#         self.pits = self.conf["primal_iterations"]

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
#                     raise NameError("CADMM primal optimizer is unknown.")

#     def primal_update(self, i, th_reg, k):
#         if self.conf["persistant_primal_opt"]:
#             opt = self.opts[i]
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
#                 raise NameError("DiNNO primal optimizer is unknown.")

#         for _ in range(self.pits):
#             opt.zero_grad()

#             # Model pass on the batch
#             pred_loss = self.pr.local_batch_loss(i)

#             # Get the primal variable WITH the autodiff graph attached.
#             th = torch.nn.utils.parameters_to_vector(
#                 self.pr.models[i].parameters()
#             )

#             reg = torch.sum(
#                 torch.square(torch.cdist(th.reshape(1, -1), th_reg))
#             )

#             loss = pred_loss + torch.dot(th, self.duals[i]) + self.rho * reg
#             loss.backward()
#             opt.step()

#         return

#     def train(self, profiler=None):
#         eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
#         oits = self.conf["outer_iterations"]
#         for k in range(oits):
#             if k % eval_every == 0 or k == oits - 1:
#                 self.pr.evaluate_metrics(at_end=(k == oits - 1))

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

#                 self.duals[i] += self.rho * torch.sum(ths[i] - thj, dim=0)
#                 th_reg = (thj + ths[i]) / 2.0
#                 self.primal_update(i, th_reg, k)

#             if profiler is not None:
#                 profiler.step()

#         return


# import torch
# import math

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
#             self.primal_lr = torch.linspace(
#                 self.conf["primal_lr_start"],
#                 self.conf["primal_lr_finish"],
#                 self.conf["outer_iterations"],
#             )
#         elif self.conf["lr_decay_type"] == "log":
#             self.primal_lr = torch.logspace(
#                 math.log(self.conf["primal_lr_start"], 10),
#                 math.log(self.conf["primal_lr_finish"], 10),
#                 self.conf["outer_iterations"],
#             )
#         else:
#             raise NameError("Unknown primal learning rate decay type.")

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
#                     raise NameError("DiNNO primal optimizer is unknown.")

#         # Initialize metrics storage
#         self.metrics = []  # List to store metrics per epoch

#     def primal_update(self, i, th_reg, k):
#         if self.conf["persistant_primal_opt"]:
#             opt = self.opts[i]
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
#                 raise NameError("DiNNO primal optimizer is unknown.")

#         for _ in range(self.pits):
#             opt.zero_grad()

#             # Model pass on the batch
#             pred_loss = self.pr.local_batch_loss(i)

#             # Get the primal variable WITH the autodiff graph attached.
#             th = torch.nn.utils.parameters_to_vector(
#                 self.pr.models[i].parameters()
#             )

#             reg = torch.sum(
#                 torch.square(torch.cdist(th.reshape(1, -1), th_reg))
#             )

#             loss = pred_loss + torch.dot(th, self.duals[i]) + self.rho * reg
#             loss.backward()
#             opt.step()

#         return

#     def train(self, profiler=None):
#         eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
#         oits = self.conf["outer_iterations"]
#         for k in range(oits):
#             if k % eval_every == 0 or k == oits - 1:
#                 # Evaluate metrics and append to self.metrics
#                 metrics = self.pr.evaluate_metrics(at_end=(k == oits - 1))
#                 self.metrics.append(metrics)  # Store metrics

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

#                 self.duals[i] += self.rho * torch.sum(ths[i] - thj, dim=0)
#                 th_reg = (thj + ths[i]) / 2.0
#                 self.primal_update(i, th_reg, k)

#             if profiler is not None:
#                 profiler.step()

#         return

import torch
import math

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
            raise NameError("Unknown primal learning rate decay type.")

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
                    raise NameError("DiNNO primal optimizer is unknown.")

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
                raise NameError("DiNNO primal optimizer is unknown.")

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
                # Update local model weights
                new_weights = (current_weights + avg_weights) / 2.0
                torch.nn.utils.vector_to_parameters(new_weights, self.pr.models[i].parameters())

    def train(self, profiler=None):
        eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
        oits = self.conf["outer_iterations"]
        for k in range(oits):
            if k % eval_every == 0 or k == oits - 1:
                # Evaluate metrics and append to self.metrics
                metrics = self.pr.evaluate_metrics(at_end=(k == oits - 1))
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
