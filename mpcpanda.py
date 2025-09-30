import torch
import numpy as np
import sys
import pinocchio as pin

sys.path.insert(0, 'mpc.pytorch')
from mpc.mpc import MPC, QuadCost, GradMethods, LinDx

class PinocchioPandaDynamics(torch.nn.Module):
    """Pinocchio-based dynamics for fixed-base Panda with analytic Jacobians.

    Uses ABA forward dynamics and computeABADerivatives for exact partials.
    Discretization: semi-implicit Euler
        v' = v + dt * a(q, v, u)
        q' = integrate(q, dt * v')   (approx Jacobians treat integrate as q + dt*v')

    Notes
    - Pinocchio runs on CPU; tensors are copied to CPU for dynamics, results
      are returned on the input device. Batch is processed in a Python loop.
    - For speed: reuse model/data and avoid allocations where possible.
    """
    def __init__(self, urdf_path, dt=0.01, device=None, with_gravity=True):
        super().__init__()
        self.dt = float(dt)
        self.device = torch.device(device) if device is not None else torch.device('cpu')

        self.pin = pin

        # Build model and data
        self.model = self.pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        if with_gravity:
            self.model.gravity.linear = np.array([0.0, 0.0, -9.81])
        else:
            self.model.gravity.linear = np.array([0.0, 0.0, 0.0])

        # Sizes (generalized for any actuated DoF)
        self.nq = self.model.nq
        self.nv = self.model.nv

        # Joint limits and efforts (from Pinocchio model)
        q_lower = torch.tensor(self.model.lowerPositionLimit, dtype=torch.float64)
        q_upper = torch.tensor(self.model.upperPositionLimit, dtype=torch.float64)
        effort_limit = torch.tensor(self.model.effortLimit, dtype=torch.float64)
        self.register_buffer('q_lower', q_lower.to(self.device))
        self.register_buffer('q_upper', q_upper.to(self.device))
        self.register_buffer('effort_limit', effort_limit.to(self.device))

    def _to_numpy(self, t):
        return t.detach().cpu().numpy()

    def _forward_single(self, q_np, v_np, u_np):
        pin = self.pin
        # Compute forward dynamics and semi-implicit integration
        a = pin.aba(self.model, self.data, q_np, v_np, u_np)  # shape (7,)
        v_next = v_np + self.dt * a
        # Simple explicit Euler for configuration update
        q_next = q_np + self.dt * v_next
        return q_next, v_next

    def forward(self, x, u):
        assert x.ndimension() == 2
        assert u.ndimension() == 2
        B = x.shape[0]
        assert u.shape[0] == B
        dev = x.device
        dtype = x.dtype

        n = self.nv
        assert x.shape[1] == 2 * n
        assert u.shape[1] == n
        q = x[:, :n]
        v = x[:, n:]

        q_next_list = []
        v_next_list = []
        for i in range(B):
            q_np = self._to_numpy(q[i]).astype(np.float64)
            v_np = self._to_numpy(v[i]).astype(np.float64)
            u_np = self._to_numpy(u[i]).astype(np.float64)
            qn, vn = self._forward_single(q_np, v_np, u_np)
            q_next_list.append(torch.from_numpy(np.asarray(qn)).to(device=dev, dtype=dtype))
            v_next_list.append(torch.from_numpy(np.asarray(vn)).to(device=dev, dtype=dtype))

        q_next = torch.stack(q_next_list, dim=0)
        v_next = torch.stack(v_next_list, dim=0)
        return torch.cat([q_next, v_next], dim=-1)

    @torch.no_grad()
    def grad_input(self, x, u):
        assert x.ndimension() == 2
        assert u.ndimension() == 2
        B = x.shape[0]
        assert u.shape[0] == B
        dev = x.device
        dtype = x.dtype
        dt = self.dt

        pin = self.pin
        n = self.nv
        assert x.shape[1] == 2 * n
        assert u.shape[1] == n
        In = torch.eye(n, device=dev, dtype=dtype).expand(B, n, n)

        R = torch.zeros(B, 2*n, 2*n, device=dev, dtype=dtype)
        S = torch.zeros(B, 2*n, n, device=dev, dtype=dtype)

        q = x[:, :n]
        v = x[:, n:]

        for i in range(B):
            q_np = self._to_numpy(q[i]).astype(np.float64)
            v_np = self._to_numpy(v[i]).astype(np.float64)
            u_np = self._to_numpy(u[i]).astype(np.float64)

            # Compute derivatives of acceleration: dadq, dadv, dadu in R^{n x n}
            d_dq, d_dv, d_du = pin.computeABADerivatives(self.model, self.data, q_np, v_np, u_np)
            # Convert to torch
            dadq = torch.from_numpy(np.asarray(d_dq)).to(device=dev, dtype=dtype)
            dadv = torch.from_numpy(np.asarray(d_dv)).to(device=dev, dtype=dtype)
            dadu = torch.from_numpy(np.asarray(d_du)).to(device=dev, dtype=dtype)

            # Semi-implicit Euler discrete-time Jacobians
            # v' = v + dt*a  => dv'/dq = dt*dadq, dv'/dv = I + dt*dadv, dv'/du = dt*dadu
            R22 = In[i] + dt * dadv
            R21 = dt * dadq
            S_bot = dt * dadu

            # Explicit Euler for position: q' = q + dt*v'
            R11 = In[i] + dt * R21
            R12 = dt * R22
            S_top = dt * S_bot

            # Assemble into R and S
            # Top rows correspond to q'
            R[i, 0:n, 0:n] = R11
            R[i, 0:n, n:2*n] = R12
            # Bottom rows correspond to v'
            R[i, n:2*n, 0:n] = R21
            R[i, n:2*n, n:2*n] = R22

            S[i, 0:n, :] = S_top
            S[i, n:2*n, :] = S_bot

        return R, S





def build_diagonal_cost_vectorized(
    x_batch: torch.Tensor,
    T: int,
    diag_C_flat: torch.Tensor,
    c_flat: torch.Tensor,
    dynamics: PinocchioPandaDynamics,
    u_weight: torch.Tensor,
):
    """Vectorized QuadCost using flattened diagonal C and linear c terms.

    Cost per step: 0.5 * x^T * diag(exp(diag_C[t])) * x + c[t]^T * x + 0.5*u_w*||u||^2

    Args:
      x_batch: [B, n_state]
      T: horizon length
      diag_C_flat: flattened log-space diagonal of C matrix, shape [n_state * T]
      c_flat: flattened linear term, shape [n_state * T]
      u_weight: scalar tensor for control cost
    Returns QuadCost with C: [T, B, n_tau, n_tau], c: [T, B, n_tau]
    """
    assert x_batch.ndimension() == 2
    B, n_state = x_batch.shape
    device, dtype = x_batch.device, x_batch.dtype

    n_ctrl = dynamics.nv
    assert n_state == dynamics.nq + dynamics.nv
    assert diag_C_flat.numel() == n_state * T
    assert c_flat.numel() == n_state * T
    assert u_weight.ndimension() == 0

    # Reshape flattened inputs to [T, n_state] and log normalize diag_C
    diag_C = torch.exp(diag_C_flat.view(T, n_state)).to(device=device, dtype=dtype)
    c_vec = c_flat.view(T, n_state).to(device=device, dtype=dtype)

    n_tau = n_state + n_ctrl
    
    # Fully vectorized construction
    # Create C matrices: [T, B, n_tau, n_tau]
    C_seq = torch.zeros(T, B, n_tau, n_tau, device=device, dtype=dtype)
    
    # State diagonal blocks: vectorized across all T and B
    # Create diagonal matrices for all timesteps at once
    diag_matrices = torch.diag_embed(diag_C)  # [T, n_state, n_state]
    C_seq[:, :, :n_state, :n_state] = diag_matrices.unsqueeze(1).expand(-1, B, -1, -1)
    
    # Control block - same for all batches and timesteps
    u_eye = u_weight * torch.eye(n_ctrl, device=device, dtype=dtype)
    C_seq[:, :, n_state:, n_state:] = u_eye.unsqueeze(0).unsqueeze(0).expand(T, B, -1, -1)

    # Create c vectors: [T, B, n_tau]
    c_seq = torch.zeros(T, B, n_tau, device=device, dtype=dtype)
    
    # State part: broadcast c_vec across batch dimension
    c_seq[:, :, :n_state] = c_vec.unsqueeze(1).expand(-1, B, -1)
    # Control part remains zero

    return QuadCost(C_seq, c_seq)


class PandaMPCLayer(torch.nn.Module):
    """Differentiable MPC layer for Panda with flattened diagonal cost terms.

    - Bakes in `dynamics` and horizon `T` at init time.
    - Accepts flattened diagonal C and linear c terms for vectorized cost construction.
    - Builds a time-varying quadratic cost: 0.5 * x^T * diag(C[t]) * x + c[t]^T * x

    Forward inputs
      x_init: [B, n_state]
      diag_C_flat: [n_state * T] flattened log-space diagonal of cost matrix over trajectory
      c_flat: [n_state * T] flattened linear cost term over trajectory
      u_weight: scalar tensor for control cost

    Returns
      x_traj: [T, B, n_state]
      u_traj: [T, B, n_ctrl]
      costs: [B]
    """
    def __init__(
        self,
        urdf_path: str,
        T: int,
        dt: float = 0.01,
        device=None,
        with_gravity: bool = True,
        lqr_iter: int = 1,
        eps: float = 1e-2,
        verbose: int = 0,
    ):
        super().__init__()
        # Initialize internal dynamics
        self.dynamics = PinocchioPandaDynamics(
            urdf_path=urdf_path, dt=dt, device=device, with_gravity=with_gravity
        )
        self.T = int(T)
        self.n_state = self.dynamics.nq + self.dynamics.nv
        self.n_ctrl = self.dynamics.nv

        self.register_buffer('rollout_step', torch.zeros((), dtype=torch.long))
        effort_max = effort_min = None

        self.mpc = MPC(
            n_state=self.n_state,
            n_ctrl=self.n_ctrl,
            T=self.T,
            u_lower=effort_min,
            u_upper=effort_max,
            lqr_iter=lqr_iter,
            grad_method=GradMethods.AUTO_DIFF,
            verbose=verbose,
            eps=eps,
            n_batch=None,  # infer from cost
            exit_unconverged=False,
            detach_unconverged=False,
        )

        self.prev_u = None

    def forward(
        self,
        x_init: torch.Tensor,
        diag_C_flat: torch.Tensor,
        c_flat: torch.Tensor,
        u_weight: torch.Tensor,
    ):
        assert x_init.ndimension() == 2 and x_init.size(1) == self.n_state
        B = x_init.size(0)
        # diag_C_flat (log-space) and c_flat should be flattened (n_state * T,) vectors
        assert diag_C_flat.ndimension() == 1 and diag_C_flat.numel() == self.n_state * self.T
        assert c_flat.ndimension() == 1 and c_flat.numel() == self.n_state * self.T
        assert u_weight.ndimension() == 0

        cost = build_diagonal_cost_vectorized(
            x_batch=x_init,
            T=self.T,
            diag_C_flat=diag_C_flat,
            c_flat=c_flat,
            dynamics=self.dynamics,
            u_weight=u_weight,
        )

        lin_dx = self._build_linear_dynamics(x_init, B)

        x_traj, u_traj, costs = self.mpc(x_init, cost, lin_dx)
        return x_traj.transpose(0, 1), u_traj.transpose(0, 1), costs.detach()

    def reset_schedule(self, step: int = 0):
        """Reset internal rollout step (e.g., at episode start)."""
        self.rollout_step = torch.tensor(int(step), dtype=torch.long, device=self.rollout_step.device)
        self.prev_u = None

    def _build_linear_dynamics(
        self,
        x_init: torch.Tensor,
        batch_size: int,
    ) -> LinDx:
        device = x_init.device
        dtype = x_init.dtype

        with torch.no_grad():
            nominal_u = torch.zeros(self.T, batch_size, self.n_ctrl, device=device, dtype=dtype)

            n = self.n_ctrl
            q0 = x_init[:, :n]
            v0 = x_init[:, n:]

            q_nominal = torch.zeros(self.T, batch_size, n, device=device, dtype=dtype)
            v_nominal = torch.zeros_like(q_nominal)

            q_nominal[0] = q0
            v_nominal[0] = v0

            # Simple nominal trajectory: stay at initial position
            for t in range(1, self.T):
                q_nominal[t] = q0

            dt = float(self.dynamics.dt)
            for t in range(1, self.T):
                v_nominal[t] = (q_nominal[t] - q_nominal[t - 1]) / dt

            x_nominal = torch.cat([q_nominal, v_nominal], dim=2)

            F_list = []
            f_list = []
            for t in range(self.T - 1):
                R, S = self.dynamics.grad_input(x_nominal[t], nominal_u[t])
                F_t = torch.cat([R, S], dim=2)
                x_pred = torch.bmm(R, x_nominal[t].unsqueeze(2)).squeeze(2)
                u_pred = torch.bmm(S, nominal_u[t].unsqueeze(2)).squeeze(2)
                f_t = x_nominal[t + 1] - x_pred - u_pred
                F_list.append(F_t)
                f_list.append(f_t)

            F = torch.stack(F_list, dim=0)
            f = torch.stack(f_list, dim=0)

        return LinDx(F, f)
