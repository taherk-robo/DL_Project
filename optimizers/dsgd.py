import torch
from utils import graph_generation
import math


class DSGD:
    def __init__(self, ddl_problem, device, conf):
        self.pr = ddl_problem
        self.conf = conf
        self.device = device

        # Get list of all model parameter pointers
        self.plists = {
            i: list(self.pr.models[i].parameters()) for i in range(self.pr.N)
        }

        # Useful numbers
        self.num_params = len(self.plists[0])
        self.alph0 = conf["alpha0"]
        self.mu = conf["mu"]

    def train(self, profiler=None):
        eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
        oits = self.conf["outer_iterations"]

        # Optimization loop
        alph = self.alph0
        for k in range(oits):
            if k % eval_every == 0 or k == oits - 1:
                self.pr.evaluate_metrics(at_end=(k == oits - 1))

            W = graph_generation.get_metropolis(self.pr.graph)
            W = W.to(self.device)
            alph = alph * (1 - self.mu * alph)

            # Iterate over the agents for communication step
            for i in range(self.pr.N):
                neighs = list(self.pr.graph.neighbors(i))
                with torch.no_grad():
                    # Update each parameter individually across all neighbors
                    for p in range(self.num_params):
                        # Ego update
                        self.plists[i][p].multiply_(W[i, i])
                        # Neighbor updates
                        for j in neighs:
                            self.plists[i][p].add_(W[i, j] * self.plists[j][p])

            # Compute the batch loss and update using the gradients
            for i in range(self.pr.N):
                # Batch loss
                bloss = self.pr.local_batch_loss(i)
                bloss.backward()

                # Locally update model with gradient
                with torch.no_grad():
                    for p in range(self.num_params):
                        self.plists[i][p].add_(-alph * self.plists[i][p].grad)
                        self.plists[i][p].grad.zero_()

            if profiler is not None:
                profiler.step()
        return
