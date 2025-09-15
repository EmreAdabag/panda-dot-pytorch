#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'mpc.pytorch'))

from mpcpanda import PandaEETrackingMPCLayer
from mpc.mpc import GradMethods


def scalar_loss_from_mpc(x_seq, u_seq):
    return 0.5 * (x_seq.pow(2).mean() + 1e-2 * u_seq.pow(2).mean())


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device('cpu')

    # Dynamics
    urdf_path = os.path.join(REPO_ROOT, 'robot_description', 'panda_with_gripper.urdf')
    dt = 0.01
    # MPC layer
    T = 10
    goal_ts_abs = torch.tensor([3, 9], dtype=torch.long, device=device)
    layer = PandaEETrackingMPCLayer(
        urdf_path=urdf_path,
        T=T,
        goal_timesteps_abs=goal_ts_abs,
        dt=dt,
        device=device,
        with_gravity=True,
        lqr_iter=2,
        verbose=0,
        eps=1e-3,
    ).to(device)

    n = layer.n_ctrl
    n_state = layer.n_state

    # Joint-space multi-goal parameters (runtime, differentiable)
    K = 2
    torch.manual_seed(0)
    np.random.seed(0)
    # Initialize joint goals near zero with small random offsets
    joint_goals = (0.2 * torch.randn(K, n, dtype=torch.float64, device=device) 
                   ).requires_grad_()

    # Initial state
    q0 = torch.zeros(n, device=device, dtype=torch.float64)
    v0 = torch.zeros(n, device=device, dtype=torch.float64)
    x_init = torch.cat([q0, v0], dim=0)[None, :]  # [1, n_state]

    # Parameters to differentiate (weights + joint goals)
    q_w = torch.tensor(5.0, device=device, dtype=torch.float64, requires_grad=True)
    v_w = torch.tensor(1e-2, device=device, dtype=torch.float64, requires_grad=True)
    u_w = torch.tensor(1e-9, device=device, dtype=torch.float64, requires_grad=True)

    # Forward solve wrapper to keep graph
    def forward_and_loss():
        # Ensure consistent goal schedule across evaluations (avoid internal step drift)
        layer.reset_schedule(0)
        x_mpc, u_mpc, _ = layer(
            x_init,
            joint_goals.unsqueeze(0),  # [1, K, n]
            q_w,
            v_w,
            u_w,
        )
        return scalar_loss_from_mpc(x_mpc, u_mpc)
    # Console report: autograd vs finite differences for joint goals
    print("Checking gradients wrt joint_goals (central FD vs autograd):")
    h_q = 1e-6
    loss = forward_and_loss()
    g_qgoals = torch.autograd.grad(loss, joint_goals, retain_graph=True, create_graph=False)[0]
    for k in range(joint_goals.size(0)):
        for i in range(n):
            base = joint_goals[k, i].item()
            with torch.no_grad():
                joint_goals[k, i] = base + h_q
            lp = forward_and_loss().item()
            with torch.no_grad():
                joint_goals[k, i] = base - h_q
            lm = forward_and_loss().item()
            with torch.no_grad():
                joint_goals[k, i] = base
            fd = (lp - lm) / (2*h_q)
            ag = g_qgoals[k, i].item()
            rel = abs(ag - fd) / (abs(fd) + 1e-12)
            print(f"  joint_goals[{k}][{i}]: autograd={ag:.6e}  FD={fd:.6e}  rel_err={rel:.2e}")

    # Weights gradients
    print("\nChecking gradients wrt weights (central FD vs autograd):")
    def check_weight(name, param, h):
        loss = forward_and_loss()
        ag = torch.autograd.grad(loss, param, retain_graph=False, create_graph=False)[0].item()
        base = param.item()
        with torch.no_grad():
            param.copy_(torch.tensor(base + h, dtype=param.dtype, device=param.device))
        lp = forward_and_loss().item()
        with torch.no_grad():
            param.copy_(torch.tensor(base - h, dtype=param.dtype, device=param.device))
        lm = forward_and_loss().item()
        with torch.no_grad():
            param.copy_(torch.tensor(base, dtype=param.dtype, device=param.device))
        fd = (lp - lm) / (2*h)
        rel = abs(ag - fd) / (abs(fd) + 1e-12)
        print(f"  {name}: autograd={ag:.6e}  FD={fd:.6e}  rel_err={rel:.2e}")

    check_weight('q_w', q_w, 1e-6)
    check_weight('v_w', v_w, 1e-6)
    check_weight('u_w', u_w, 1e-12)


if __name__ == '__main__':
    main()
