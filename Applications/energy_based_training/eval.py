import argparse
import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import mdtraj
import numpy as np
import torch
from tqdm import tqdm

from egnn import EGNN_dynamics_AD2, remove_mean
from energy.aldp import AldpBoltzmann


def evaluate(x, target, path):
    x_np = x.detach().cpu().numpy() / 5
    aldp = target.system
    topology = mdtraj.Topology.from_openmm(aldp.topology)
    test_traj = mdtraj.Trajectory(x_np.reshape(-1, 22, 3), topology)
    psi_d = mdtraj.compute_psi(test_traj)[1].reshape(-1)
    phi_d = mdtraj.compute_phi(test_traj)[1].reshape(-1)
    is_nan = np.logical_or(np.isnan(psi_d), np.isnan(phi_d))
    not_nan = np.logical_not(is_nan)
    psi_d = psi_d[not_nan]
    phi_d = phi_d[not_nan]

    plt.figure(figsize=(3, 3))
    plt.hist2d(
        phi_d,
        psi_d,
        bins=64,
        norm=mpl.colors.LogNorm(),
        range=[[-np.pi, np.pi], [-np.pi, np.pi]],
    )
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(r"$\phi$", fontsize=24)
    plt.ylabel(r"$\psi$", fontsize=24)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def load_model(checkpoint, data_sigma, device):
    ema_net = EGNN_dynamics_AD2(
        n_particles=22,
        n_dimension=3,
        hidden_nf=256,
        device=device,
        act_fn=torch.nn.SiLU(),
        n_layers=5,
        recurrent=True,
        attention=True,
        condition_time=True,
        tanh=True,
        mode="egnn_dynamics",
        agg="sum",
        data_sigma=data_sigma,
    ).to(device).requires_grad_(False)
    ema_net.load_state_dict(torch.load(checkpoint, map_location=device))
    return ema_net


def make_time_grid(tmax=20.0, tmin=1e-6, rho=7, steps=200):
    ts = tmin ** (1 / rho) + np.arange(steps) / (steps - 1) * (
        tmax ** (1 / rho) - tmin ** (1 / rho)
    )
    return ts**rho


def EM_solve(model, start_samples, ts, n_particles):
    with torch.no_grad():
        samples = start_samples
        for i in range(ts.shape[0] - 1, 0, -1):
            t = torch.ones(samples.shape[0], 1, device=samples.device) * ts[i]
            t_1 = torch.ones(samples.shape[0], 1, device=samples.device) * ts[i - 1]
            Delta_t = (t - t_1).abs()
            score = model(samples, t.squeeze(-1))
            std = torch.sqrt(2 * Delta_t * t)
            dx = score * 2 * t * Delta_t + std * remove_mean(
                torch.randn_like(samples), n_particles, 3
            )
            samples = samples + dx
        return samples


class LangevinDynamics:
    def __init__(self, x, energy_func, step_size, mh=True, device="cpu"):
        self.x = x
        self.step_size = step_size
        self.energy_func = energy_func
        self.mh = mh
        self.device = device

        if self.mh:
            x_c = self.x.detach()
            x_c.requires_grad = True
            f_xc = self.energy_func(x_c)
            grad_xc = torch.autograd.grad(f_xc.sum(), x_c, create_graph=False)[0]
            self.f_x = f_xc.detach()
            self.grad_x = grad_xc.detach()

    def sample(self):
        if not self.mh:
            x_c = self.x.detach()
            x_c.requires_grad = True
            f_xc = self.energy_func(x_c)
            grad_xc = torch.autograd.grad(f_xc.sum(), x_c, create_graph=False)[0]
            x_p = (
                x_c
                - self.step_size * grad_xc
                + torch.sqrt(torch.tensor(2.0 * self.step_size, device=self.device))
                * torch.randn_like(x_c, device=self.device)
            )
            self.x = x_p.detach()
            return copy.deepcopy(x_p.detach()), f_xc.detach()

        x_c = self.x.detach()
        f_xc = self.f_x.detach()
        grad_xc = self.grad_x.detach()

        x_p = (
            x_c
            - self.step_size * grad_xc
            + torch.sqrt(torch.tensor(2.0 * self.step_size, device=self.device))
            * torch.randn_like(self.x, device=self.device)
        )
        x_p = x_p.detach()
        x_p.requires_grad = True
        f_xp = self.energy_func(x_p)
        grad_xp = torch.autograd.grad(f_xp.sum(), x_p, create_graph=False)[0]

        log_joint_prob_2 = -f_xc - torch.norm(
            x_p - x_c + self.step_size * grad_xc, dim=-1
        ) ** 2 / (4 * self.step_size)
        log_joint_prob_1 = -f_xp - torch.norm(
            x_c - x_p + self.step_size * grad_xp, dim=-1
        ) ** 2 / (4 * self.step_size)

        log_accept_rate = log_joint_prob_1 - log_joint_prob_2
        is_accept = torch.rand_like(log_accept_rate).log() <= log_accept_rate
        is_accept = is_accept.unsqueeze(-1)

        self.x = torch.where(is_accept, x_p.detach(), self.x)
        self.f_x = torch.where(is_accept.squeeze(-1), f_xp.detach(), self.f_x)
        self.grad_x = torch.where(is_accept, grad_xp.detach(), self.grad_x)

        acc_rate = torch.minimum(
            torch.ones_like(log_accept_rate), log_accept_rate.exp()
        ).mean()
        return copy.deepcopy(self.x.detach()), acc_rate.item()


def run_diffusion(args, data, target, n_particles, n_batches):
    ema_net = load_model(args.checkpoint, data.std().item(), args.device)
    ts = make_time_grid(args.tmax, args.tmin, args.rho, args.steps)

    with torch.no_grad():
        samples = []
        for _ in tqdm(range(n_batches)):
            _samples = EM_solve(
                ema_net,
                remove_mean(
                    torch.randn(args.batch_size, 3 * n_particles, device=args.device),
                    n_particles,
                    3,
                )
                * ts[-1],
                ts,
                n_particles,
            )
            samples.append(_samples)
        samples = torch.cat(samples, dim=0)

    evaluate(samples, target, args.plot_path)
    np.save(args.save_path, samples.detach().cpu().numpy().reshape(-1, 22, 3))
    print(f"saved samples to {args.save_path}, plot to {args.plot_path}")


def run_mcmc(args, data, target):
    ema_net = load_model(args.checkpoint, data.std().item(), args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    step_size = args.step_size
    x = data[: args.batch_size].clone()
    energy = lambda x: -ema_net.logp(x, torch.zeros(x.shape[0], device=args.device)).flatten()

    samples = []
    for s in tqdm(range(args.num_samples*args.save_every//args.batch_size)):
        x, acc = LangevinDynamics(x, energy, step_size, True, args.device).sample()
        if acc >= 0.7:
            step_size *= 1.5
        if acc <= 0.5:
            step_size /= 1.5

        if s % args.save_every == 0:
            samples.append(x.detach().cpu())

        if s % args.eval_every == 0 or s == args.num_samples*args.save_every - 1:
            try:
                _samples = torch.cat(samples, dim=0)
                n_plot = min(args.n_plot, _samples.shape[0])
                plot_idx = np.random.choice(_samples.shape[0], n_plot, replace=False)
                evaluate(_samples[plot_idx], target, args.plot_path)
                torch.save(_samples, args.save_path)
                print(
                    f"step {s}: acc={acc:.3f}, step_size={step_size:.3g}, "
                    f"saved {args.save_path}"
                )
            except Exception as e:
                print(f"eval failed at step {s}: {e}")

    _samples = torch.cat(samples, dim=0)
    torch.save(_samples, args.save_path)
    print(f"saved MCMC samples to {args.save_path}")


def main(args):
    n_particles = 22
    target = AldpBoltzmann(300, "implicit", device=args.device)
    data = (
        torch.from_numpy(remove_mean(mdtraj.load(args.data).xyz, 22, 3))
        .to(args.device)
        .reshape(-1, 22 * 3)
        * 5
    )

    if args.mode == "diffusion":
        run_diffusion(args, data, target, n_particles, n_batches=args.num_samples//args.batch_size)
    elif args.mode == "mcmc":
        run_mcmc(args, data, target)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate energy-based ALDP models (diffusion sampling or Langevin MCMC)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="mcmc",
        choices=["diffusion", "mcmc"],
        help="diffusion: reverse SDE sampling; mcmc: Langevin on EBM energy",
    )
    parser.add_argument("--data", type=str, default="trajectory300.h5")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint", type=str, default="ema_net1e3.pt")
    parser.add_argument("--plot_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)

    # diffusion sampling
    parser.add_argument("--tmax", type=float, default=20.0)
    parser.add_argument("--tmin", type=float, default=1e-6)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=200)

    # Langevin MCMC
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--step_size", type=float, default=6e-2)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--n_plot", type=int, default=50000)

    args = parser.parse_args()

    if args.mode == "diffusion":
        args.batch_size = 1000 if args.batch_size is None else args.batch_size
        args.save_path = "dm_samples.npy" if args.save_path is None else args.save_path
        args.plot_path = "Rplot_dm.png" if args.plot_path is None else args.plot_path
    else:
        args.batch_size = 1000 if args.batch_size is None else args.batch_size
        args.save_path = "ALDP_EBM_Reg.pkl" if args.save_path is None else args.save_path
        args.plot_path = "Rplot_reg.png" if args.plot_path is None else args.plot_path

    main(args)
