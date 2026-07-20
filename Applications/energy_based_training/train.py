import argparse
from collections import OrderedDict
from copy import deepcopy

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


def log_norm_prob(x, mu, std):
    std = std.expand(x.shape)
    return -0.5 * ((x - mu) / std).pow(2).sum(-1) - (std.log() + np.log(2 * np.pi) / 2).sum(-1)


def rne_loss(x_t, models, t, Delta_t):
    t_1 = t + Delta_t
    t = t.reshape(-1, 1)
    t_1 = t_1.reshape(-1, 1)

    fwd_mean = x_t
    fwd_std = torch.sqrt(2 * Delta_t * t)
    x_t1 = x_t + fwd_std * torch.randn_like(x_t)

    x_t1 = x_t1.detach()
    x_t = x_t.detach()

    dm_fwd = log_norm_prob(x_t1, fwd_mean, fwd_std)

    score = models.forward(x_t1, t_1.squeeze())
    bwd_mean = x_t1 + score * 2 * t_1 * Delta_t
    bwd_std = torch.sqrt(2 * Delta_t * t_1)
    dm_bwd = log_norm_prob(x_t, bwd_mean, bwd_std)

    ref_std = lambda time: (1**2 + time**2) ** 0.5
    ref_score = lambda x, time: -x / ref_std(time) ** 2

    fwd_mean = x_t
    fwd_std = torch.sqrt(2 * Delta_t * t)
    ref_fwd = log_norm_prob(x_t, 0, ref_std(t)) + log_norm_prob(x_t1, fwd_mean, fwd_std)

    bwd_mean = x_t1 + ref_score(x_t1, t_1) * 2 * t_1 * Delta_t
    bwd_std = torch.sqrt(2 * Delta_t * t_1)
    ref_bwd = log_norm_prob(x_t1, 0, ref_std(t_1)) + log_norm_prob(x_t, bwd_mean, bwd_std)

    pt = models.logp(x_t, t.squeeze()).flatten()
    pt1 = models.logp(x_t1, t_1.squeeze()).flatten()

    R = (pt + dm_fwd.detach() - dm_bwd.detach() - ref_fwd.detach() + ref_bwd.detach() - pt1) ** 2
    return R.mean()


def DSM_loss(x_t, models, x0, t):
    score = models(x_t, t)
    x_hat = score * t[:, None] ** 2 + x_t
    return (((x0 - x_hat) ** 2).sum(-1) / t**2).mean()


def update_sg_model(mu, online_model, sg_model):
    with torch.no_grad():
        online_params = OrderedDict(online_model.named_parameters())
        sg_params = OrderedDict(sg_model.named_parameters())
        assert online_params.keys() == sg_params.keys()

        for name, param in online_params.items():
            sg_params[name].sub_((1.0 - mu) * (sg_params[name] - param))
            sg_params[name].requires_grad_(False)

        online_buffers = OrderedDict(online_model.named_buffers())
        sg_buffers = OrderedDict(sg_model.named_buffers())
        assert online_buffers.keys() == sg_buffers.keys()

        for name, buffer in online_buffers.items():
            sg_buffers[name].copy_(buffer)


def get_dist(x):
    x = (((x.reshape(-1, 22, 1, 3) - x.reshape(-1, 1, 22, 3)) ** 2).sum(-1).sqrt()).cpu()
    diagx = torch.triu_indices(x.shape[1], x.shape[1], 1)
    return x[:, diagx[0], diagx[1]].flatten()


def EM_solve(model, start_samples, ts, n_particles):
    with torch.no_grad():
        samples = start_samples
        for i in tqdm(range(ts.shape[0] - 1, 0, -1)):
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


def main(args):
    device = args.device
    n_particles = 22
    target = AldpBoltzmann(300, "implicit")

    data = (
        torch.from_numpy(remove_mean(mdtraj.load(args.data).xyz, 22, 3))
        .to(device)
        .reshape(-1, 22 * 3)
        * 5
    )

    denoising_net = EGNN_dynamics_AD2(
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
        data_sigma=data.std().item(),
    ).to(device)

    if args.resume is not None:
        denoising_net.load_state_dict(torch.load(args.resume, map_location=device))

    opt = torch.optim.Adam(denoising_net.parameters(), lr=args.lr)
    ema_net = deepcopy(denoising_net).to(device)

    LOSS = []
    RNE_LOSS = []

    for epoch in tqdm(range(args.n_epoch)):
        x = remove_mean(
            data[torch.randint(0, data.shape[0], (args.batch_size,))].to(device),
            n_particles,
            3,
        )

        logt = torch.randn(args.batch_size, device=device) * 1.2 - 1.2
        noises = remove_mean(torch.randn_like(x), n_particles, 3)
        x_t = remove_mean(x + noises * logt.exp()[:, None], n_particles, 3)

        rne = rne_loss(x_t, denoising_net, logt.exp(), args.delta_t)
        loss = DSM_loss(x_t, denoising_net, x, logt.exp()) + args.rne_weight * rne
        RNE_LOSS.append(rne.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoising_net.parameters(), 5.0)
        opt.step()
        opt.zero_grad()
        LOSS.append(loss.item())

        update_sg_model(args.ema_decay, denoising_net, ema_net)

        if (epoch + 1) % args.eval_every == 0 or epoch == 0:
            tmax = 20
            tmin = 1e-6
            rho = 7
            steps = 200
            ts = tmin ** (1 / rho) + np.arange(steps) / (steps - 1) * (
                tmax ** (1 / rho) - tmin ** (1 / rho)
            )
            ts = ts**rho

            with torch.no_grad():
                samples = []
                for _ in range(10):
                    _samples = EM_solve(
                        ema_net,
                        remove_mean(
                            torch.randn(50, 3 * n_particles, device=device),
                            n_particles,
                            3,
                        )
                        * ts[-1],
                        ts,
                        n_particles,
                    )
                    samples.append(_samples)
                samples = torch.cat(samples, dim=0)

            evaluate(samples, target, f"Rplot_{epoch}.png")

            with torch.no_grad():
                plt.figure(figsize=(6, 3))
                plt.subplot(1, 2, 1)
                plt.hist(get_dist(samples[::10]), 100, density=1, alpha=1, histtype="step")
                plt.hist(get_dist(data[::100]), 100, density=1, alpha=1, histtype="step")
                plt.subplot(1, 2, 2)
                plt.hist(get_dist(data[::100]), 100, density=1, alpha=1, histtype="step")
                plt.savefig(f"Dplot_{epoch}.png", dpi=300, bbox_inches="tight")
                plt.close()

            plt.figure(figsize=(5, 3))
            plt.plot(LOSS)
            plt.yscale("log")
            plt.savefig(f"Lplot_{epoch}.png", dpi=300, bbox_inches="tight")
            plt.close()

            torch.save(ema_net.state_dict(), args.save_path)
            print(
                f"epoch {epoch}: loss={LOSS[-1]:.6g}, rne={RNE_LOSS[-1]:.6g}, "
                f"saved {args.save_path}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy-based training with RNE regularization")
    parser.add_argument("--data", type=str, default="trajectory300.h5")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_epoch", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rne_weight", type=float, default=1e3)
    parser.add_argument("--delta_t", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--save_path", type=str, default="ema_net1e3.pt")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from")
    args = parser.parse_args()
    main(args)
