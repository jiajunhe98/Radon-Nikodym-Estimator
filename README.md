# Radon-Nikodym Estimator (RNE)


[![Conference](https://img.shields.io/badge/ICLR-2026-blue)](#reference)
[![arXiv](https://img.shields.io/badge/arXiv-2506.05668-b31b1b.svg)](https://arxiv.org/abs/2506.05668)



This repo contains code for **[RNE: plug-and-play diffusion inference-time control and energy-based training](https://arxiv.org/abs/2506.05668)** (ICLR 2026). 

This repo is currently under construction, and we will release the complete code ASAP.  Sorry for the delay.

## 🗓️ Plans

- [x] Tutorial notebook for density estimator;

- [x] Tutorial notebook for diffusion control;

- [x] Tutorial notebook for energy-based training;

- [x] Annealing application;

- [ ] Reward-tilting application;

- [ ] Energy-based training application.


## 🎓 Notebook and Tutorial

To aid understanding of how to implement our method, we provide several notebooks in ```Notebooks```, covering:

(1) [density estimation](https://github.com/jiajunhe98/Radon-Nikodym-Estimator/blob/main/Notebooks/density_estimator.ipynb); 

(2) [diffusion control](https://github.com/jiajunhe98/Radon-Nikodym-Estimator/blob/main/Notebooks/diffusion_control_anneal.ipynb); 

(3) [energy-based training](https://github.com/jiajunhe98/Radon-Nikodym-Estimator/blob/main/Notebooks/energy_based_training.ipynb).

## 💻 Experiments 


```
Applications
├── anneal/      # RNE tempering
├── energy_based_training/     # Energy-based training with RNE regularization
├── stitch/     # (under construction)
└── ctmc/       # (under construction)
```

##### 👉🏻[Test-time Annealing on Boltzmann distributions](https://github.com/jiajunhe98/Radon-Nikodym-Estimator/tree/main/Applications/anneal)


##### 👉🏻Test-time Reward-tilting with CTMC


##### 👉🏻Test-time stitching for maze navigation


##### 👉🏻[Energy-based diffusion training on Boltzmann distributions (Alanine Dipeptide)](https://github.com/jiajunhe98/Radon-Nikodym-Estimator/tree/main/Applications/energy_based_training)




Please email jh2383@cam.ac.uk if you have more questions.




## 📑 Reference

If you find this repository useful, please consider citing:

```bibtex
@inproceedings{he2025rne,
  title={RNE: plug-and-play diffusion inference-time control and energy-based training},
  author={He, Jiajun and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel and Du, Yuanqi and Vargas, Francisco},
  booktitle={ICLR},
  year={2025}
}
