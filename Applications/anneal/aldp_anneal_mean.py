import torch
from energy.aldp import AldpBoltzmann
from network.egnn import EGNN_dynamics_AD2, remove_mean
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mdtraj
import matplotlib as mpl




import argparse


def main(args):

    print(args.heuristic)

    # check if fkc/heuristic has no more than 2 true
    if args.fkc + args.heuristic > 1:
        raise ValueError('Only at most one of fkc, heuristic can be true')


    dir_name = 'Scaling_Res_aldp_bsz' + str(args.bsz)
    if args.fkc == True:
        dir_name += 'fkc'
    elif args.heuristic == True:
        dir_name += 'heuristic'
    else:
        dir_name += f'_fwd{args.fwd_lamb}_bwd{args.bwd_lamb}'
        if args.ref == True:
            dir_name += '_use_ref'
        else:
            dir_name += '_no_ref'


    # make dir
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


    temp = 800/300
    n_particles = 22
    target = AldpBoltzmann(300, 'implicit')
    device = 'cuda'

    def evaluate(x, target, name):
        
        x_np = x.detach().cpu().numpy() / 5 # unscale the data
        # plot x1 and x1_hat
        aldp = target.system
        topology = mdtraj.Topology.from_openmm(aldp.topology)
        test_traj = mdtraj.Trajectory(x_np.reshape(-1, 22, 3), topology)
        psi_d = mdtraj.compute_psi(test_traj)[1].reshape(-1)
        phi_d = mdtraj.compute_phi(test_traj)[1].reshape(-1)
        is_nan = np.logical_or(np.isnan(psi_d), np.isnan(phi_d))
        not_nan = np.logical_not(is_nan)
        psi_d = psi_d[not_nan]
        phi_d = phi_d[not_nan]  

        # Ramachandran plot
        plt.figure(figsize=(3, 3))
        
        plt.hist2d(phi_d, psi_d, bins=64, norm=mpl.colors.LogNorm(),
                    range=[[-np.pi, np.pi], [-np.pi, np.pi]])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('$\phi$', fontsize=24)
        plt.ylabel('$\psi$', fontsize=24)
        plt.savefig(dir_name + '/' + name + '.png', dpi=300, bbox_inches='tight')

    def EM_solve_temper(model, start_samples, beta):
        # directly temper the score
        with torch.no_grad():
            samples = start_samples
            for i in tqdm(range(ts.shape[0]-1, 0, -1)):
                t = torch.ones(samples.shape[0], 1).to(samples.device) * ts[i]
                t_1 = torch.ones(samples.shape[0], 1).to(samples.device) * ts[i-1]

                Delta_t = (t - t_1).abs()
                x_hat = model(samples, t.squeeze(-1)) 
                std = torch.sqrt(2*Delta_t*t)
                score = - (samples - x_hat) / t ** 2 * beta

                dx = score * 2 * t * Delta_t + std * torch.randn_like(samples) 

                samples = samples + dx
            return samples
    def EM_solve_temper_fkc(model, start_samples, beta):
        # use FKC to temper 
        with torch.no_grad():
            samples = start_samples
            bsz = samples.shape[0]
            w = 0
            ESS = []
            Samples = []
            for i in tqdm(range(ts.shape[0]-1, 0, -1)):
                t = torch.ones(samples.shape[0], 1).to(samples.device) * ts[i]
                t_1 = torch.ones(samples.shape[0], 1).to(samples.device) * ts[i-1]

                Delta_t = (t - t_1).abs()
                x_hat = model(samples, t.squeeze(-1)) 
                std = torch.sqrt(2*Delta_t*t)
                score = - (samples - x_hat) / t ** 2 

                dx = score * 2 * t * Delta_t * beta + std * remove_mean(torch.randn_like(samples), n_particles, 3)

                samples_new = samples + dx

                dw = (beta - 1) * beta / 2 * (score ** 2).sum(-1) * 2 * t[:, 0] * Delta_t[:, 0]

                w += dw
                
                # calculate ESS
                _w = torch.nn.Softmax(0)(w)
                ess = _w.sum() ** 2 / (_w ** 2).sum()  
                ESS.append(ess.item())

                # resample
                # clip w
                if ess < 0.75 * bsz or i == 1:
                    _w = torch.clamp(_w, max=_w.quantile(0.999))
                    _w = _w.detach().cpu().numpy() + 1e-10
                    resample_idx = np.random.choice(bsz, (bsz, ), p=_w/_w.sum())

                    samples = samples_new[resample_idx]

                    w = 0
                else:
                    samples = samples_new

                Samples.append(samples.detach().cpu().numpy())

            return samples, ESS, Samples
    def EM_solve_temper_rnc(model, start_samples, beta, fwd_lamb, bwd_lamb, with_reference=False):
        # use RNE to temper
        with torch.no_grad():
            samples = start_samples
            bsz = samples.shape[0]
            w = 0
            acc_w = 0
            ESS = []
            Samples = []
            Resample = 0
            for i in tqdm(range(ts.shape[0]-1, 0, -1)):
                t = torch.ones(samples.shape[0], 1).to(samples.device) * ts[i]
                t_1 = torch.ones(samples.shape[0], 1).to(samples.device) * ts[i-1]

                Delta_t = (t - t_1).abs()
                x_hat = model(samples, t.squeeze(-1)) 
                std = torch.sqrt(2*Delta_t*t)
                score = - (samples - x_hat) / t ** 2 

                dx = score * 2 * t * Delta_t * beta *  bwd_lamb + std * remove_mean(torch.randn_like(samples), n_particles, 3)

                samples_new = samples + dx

                # diffusion forward process
                fwd_mean = samples_new
                fwd_std = torch.sqrt(2*Delta_t*t_1)
                dm_fwd = log_norm_prob(samples, fwd_mean, fwd_std) 

                # diffusion backward process
                bwd_mean = samples + score * 2 * t * Delta_t
                bwd_std = torch.sqrt(2*Delta_t*t)
                dm_bwd = log_norm_prob(samples_new, bwd_mean, bwd_std)


                # sampling forward process

                x_hat = model(samples_new, t_1.squeeze(-1)) 
                score_new = - (samples_new - x_hat) / t_1 ** 2 

                fwd_mean = samples_new + score_new * 2 * t_1 * Delta_t * beta * fwd_lamb
                fwd_std = torch.sqrt(2*Delta_t*t_1)
                sample_fwd = log_norm_prob(samples, fwd_mean, fwd_std) 

                # sampling backward process
                bwd_mean = samples + score * 2 * t * Delta_t * beta * bwd_lamb
                bwd_std = torch.sqrt(2*Delta_t*t)
                sample_bwd = log_norm_prob(samples_new, bwd_mean, bwd_std)

                if not with_reference:

                    # log weight
                    w = w + (sample_fwd - sample_bwd) + (dm_bwd - dm_fwd) * beta
                    acc_w = acc_w+ (sample_fwd - sample_bwd) + (dm_bwd - dm_fwd) * beta
                
                else:
                    # first define reference distribution
                    ref_std = lambda time: (1**2 + time ** 2)**0.5
                    ref_score = lambda x, time: - x / ref_std(time) ** 2
                    # ref forward process
                    fwd_mean = samples_new
                    fwd_std = torch.sqrt(2*Delta_t*t_1)
                    ref_fwd = log_norm_prob(samples_new, 0, ref_std(t_1)) + log_norm_prob(samples, fwd_mean, fwd_std) 

                    # ref backward process
                    bwd_mean = samples + ref_score(samples_new, t) * 2 * t * Delta_t
                    bwd_std = torch.sqrt(2*Delta_t*t)
                    ref_bwd = log_norm_prob(samples, 0, ref_std(t)) + log_norm_prob(samples_new, bwd_mean, bwd_std)

                    # log weight
                    w = w + (sample_fwd - ref_fwd - sample_bwd + ref_bwd) + (dm_bwd - ref_bwd - dm_fwd + ref_fwd) * beta
                    acc_w = acc_w + (sample_fwd - ref_fwd - sample_bwd + ref_bwd) + (dm_bwd - ref_bwd - dm_fwd + ref_fwd) * beta
                
                # calculate ESS
                _w = torch.nn.Softmax(0)(w)
                ess = _w.sum() ** 2 / (_w ** 2).sum()  
                ESS.append(ess.item())

                # resample
                # clip w
                if ess < 0.75 * bsz or i == 1:
                    _w = torch.clamp(_w, max=_w.quantile(0.999))
                    _w = _w.detach().cpu().numpy() + 1e-10
                    resample_idx = np.random.choice(bsz, (bsz, ), p=_w/_w.sum())
                    samples = samples_new[resample_idx]
                    w = 0
                    Resample += 1
                else:
                    samples = samples_new

                Samples.append(samples.detach().cpu().numpy())

            return samples, ESS, Samples , Resample , acc_w.var()  

    def log_norm_prob(x, mu, std):
        # expand std to match x shape
        std = std.expand(x.shape)
        return -0.5 * ((x - mu) / std).pow(2).sum(-1) - (std.log() + np.log(2*np.pi)/2).sum(-1)

    # load data
    # note that the data was scaled by 5 following FEAT paper
    data = torch.from_numpy(remove_mean(mdtraj.load('trajectory800.h5').xyz, 22, 3)).to(device).reshape(-1, 22*3) * 5
    data300 = torch.from_numpy(remove_mean(mdtraj.load('trajectory300.h5').xyz, 22, 3)).to(device).reshape(-1, 22*3)  * 5


    ema_net = EGNN_dynamics_AD2(
                    n_particles=22, n_dimension=3, hidden_nf=256, device='cuda',
                    act_fn=torch.nn.SiLU(), n_layers=5, recurrent=True, attention=True,
                    condition_time=True, tanh=True, mode='egnn_dynamics', agg='sum', data_sigma=data.std().item()).to(device).requires_grad_(False)
    ema_net.load_state_dict(torch.load('ema_net_800k.pt', map_location=device))

    tmax = 10
    tmin = 1e-3
    rho = 7
    steps = 200
    ts = tmin ** (1/rho) + np.arange(steps)/(steps-1) * (tmax ** (1/rho) - tmin ** (1/rho))
    ts = ts ** rho


    BSZ = args.bsz
    repeats = args.repeats

    for repeat in range(repeats):

        if args.heuristic == True:
            with torch.no_grad():
                samples = EM_solve_temper(ema_net, torch.randn(BSZ, 3*n_particles, device=device)*ts[-1], temp)
        elif args.fkc == True:
            with torch.no_grad():
                samples, ESS, Samples = EM_solve_temper_fkc(ema_net, torch.randn(BSZ, 3*n_particles, device=device)*ts[-1]/np.sqrt(temp), temp)
        else:
            with torch.no_grad():
                samples, ESS, Samples, Resample,  w_var  = EM_solve_temper_rnc(ema_net, torch.randn(BSZ, 3*n_particles, device=device)*ts[-1]/np.sqrt(temp), temp, 
                                                            args.fwd_lamb, args.bwd_lamb, with_reference=args.ref)

        torch.save(samples, dir_name + '/samples_%d.pt'%repeat)
        plt.hist(target.log_prob(samples[:]/5).detach().cpu(), bins=100, density=True, alpha=0.5, label='target')
        plt.hist(target.log_prob(data300[::30]/5).detach().cpu(), bins=100, density=True, alpha=0.5, label='data')
        plt.legend()
        plt.savefig(dir_name + '/log_prob_%d.png'%repeat, dpi=300, bbox_inches='tight')
        plt.close()
        evaluate(samples, target, 'Rplot_%d'%repeat)
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--bwd_lamb', type=float)
    parser.add_argument('--fwd_lamb', type=float)
    parser.add_argument('--fkc', type=int)
    parser.add_argument('--heuristic', type=int)
    parser.add_argument('--ref', type=int)
    parser.add_argument('--bsz', type=int, default=5000)
    parser.add_argument('--repeats', type=int, default=50)

    args = parser.parse_args()

    # make fkc, heuristic, ref to bool
    # if not fkc, or heuristic, then use RNE
    args.fkc = bool(args.fkc)
    args.heuristic = bool(args.heuristic)
    args.ref = bool(args.ref)
    
    main(args)
