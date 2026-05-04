"""
Export trained PyTorch models to JSON so they can run in the browser demo.

Each model is just a few linear layers — we export the weight matrices
and biases, then reimplement the forward pass in JavaScript.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import numpy as np

from src.comm.commnet import CommNet
from src.comm.ic3net import IC3Net
from src.comm.tarmac import TarMAC
from src.comm.gated_attn import GatedAttnComm


def build_comm_module(method, obs_dim, act_dim, n_agents,
                      hidden_dim=64, msg_dim=32):
    if method == 'commnet':
        return CommNet(obs_dim, hidden_dim, msg_dim, act_dim, n_agents)
    elif method == 'ic3net':
        return IC3Net(obs_dim, hidden_dim, msg_dim, act_dim, n_agents)
    elif method == 'tarmac':
        return TarMAC(obs_dim, hidden_dim, msg_dim, act_dim, n_agents, n_heads=4)
    elif method == 'gated_attn':
        return GatedAttnComm(obs_dim, hidden_dim, msg_dim, act_dim, n_agents,
                             n_heads=4, connectivity_weight=0.5)
    else:
        raise ValueError(f"Unknown method: {method}")


def export_model(method, results_dir, obs_dim, act_dim, n_agents):
    model_file = os.path.join(results_dir, f"{method}_seed0.pt")
    if not os.path.exists(model_file):
        print(f"Skipping {method} — no model at {model_file}")
        return None

    comm = build_comm_module(method, obs_dim, act_dim, n_agents)
    comm.load_state_dict(torch.load(model_file, weights_only=True))
    comm.eval()

    # Export all parameters as nested dict of lists
    params = {}
    for name, param in comm.named_parameters():
        params[name] = param.detach().cpu().numpy().tolist()

    print(f"  {method}: {len(params)} parameter tensors exported")
    return params


def main():
    obs_dim = 18   # MPE simple_spread with 3 agents
    act_dim = 5
    n_agents = 3
    results_dir = 'results'

    methods = ['commnet', 'ic3net', 'tarmac', 'gated_attn']
    all_models = {}

    for method in methods:
        params = export_model(method, results_dir, obs_dim, act_dim, n_agents)
        if params:
            all_models[method] = {
                'params': params,
                'method': method,
                'obs_dim': obs_dim,
                'act_dim': act_dim,
                'n_agents': n_agents,
                'hidden_dim': 64,
                'msg_dim': 32,
            }

    output_file = 'demo/models.json'
    with open(output_file, 'w') as f:
        json.dump(all_models, f)

    size_kb = os.path.getsize(output_file) / 1024
    print(f"\nExported to {output_file} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
