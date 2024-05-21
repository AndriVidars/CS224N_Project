import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, k):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.num_experts = num_experts
        self.k = k

    def forward(self, x):
        logits = self.fc(x)
        topk_values, _ = torch.topk(logits, self.k, dim=-1)
        min_topk = topk_values[:, -1].unsqueeze(-1).expand_as(logits)
        mask = logits < min_topk
        logits[mask] = float('-inf')
        topk_softmax = torch.softmax(logits, dim=-1)
        return topk_softmax

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.linear(x))

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=8, k=2):
        super(MoE, self).__init__()
        self.gating_network = GatingNetwork(input_dim, num_experts, k)
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.k = k

    def forward(self, x):
        gating_weights = self.gating_network(x.mean(dim=1))  # Gating based on mean-pooled input

        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Weighted sum of expert outputs based on gating weights
        weighted_expert_output = torch.sum(gating_weights.unsqueeze(-1).unsqueeze(-1) * expert_outputs, dim=1)
        
        return weighted_expert_output
