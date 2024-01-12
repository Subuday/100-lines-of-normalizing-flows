import math
import torch
from torch import nn
from tqdm import tqdm


def gaussian_log_pdf(z):
    return -0.5 * (torch.log(torch.tensor([math.pi * 2], device=z.device)) + z ** 2).sum(dim=-1)


class PlanarFlow(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(input_dim))
        self.w = nn.Parameter(torch.randn(input_dim))
        self.b = nn.Parameter(torch.randn(1))
        self.h = nn.Tanh()
        self.h_prime = lambda z: 1 - self.h(z) ** 2

    def constrained_u(self):
        wu = torch.matmul(self.w.T, self.u)
        m = lambda x: -1 + torch.log(1 + torch.exp(x))
        return self.u + (m(wu) - wu) * (self.w / (torch.norm(self.w) ** 2 + 1e-15))

    def forward(self, z):
        u = self.constrained_u()
        hidden_units = torch.matmul(self.w.T, z.T) + self.b
        x = z + u.unsqueeze(0) + self.h(hidden_units) + self.w.unsqueeze(-1)
        psi = self.h_prime(hidden_units).unsqueeze(0) * self.w.unsqueeze(-1)
        log_det = torch.log((1 + torch.matmul(u.T, psi)).abs() + 1e-15)
        return x, log_det


class NormalizingFlow(nn.Module):

    def __init__(self, length, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            *(PlanarFlow(input_dim) for _ in range(length))
        )

    def forward(self, z):
        log_jacobians = 0
        for layer in self.layers:
            z, log_jacobian = layer(z)
            log_jacobians += log_jacobian
        return z, log_jacobians


def train(flow, optimizer, nb_epochs, log_density, batch_size, data_dim, device):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        # Generate new samples from the flow
        z0 = torch.randn(batch_size, data_dim).to(device)
        zk, log_jacobian = flow(z0)

        # Evaluate the exact and approximated densities
        flow_log_density = gaussian_log_pdf(z0) - log_jacobian
        exact_log_density = log_density(zk).to(device)

        # Compute the loss
        reverse_kl_divergence = (flow_log_density - exact_log_density).mean()
        optimizer.zero_grad()
        loss = reverse_kl_divergence
        loss.backward()
        optimizer.step()

        training_loss.append(loss.item())
    return training_loss
