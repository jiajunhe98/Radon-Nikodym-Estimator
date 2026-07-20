# Energy-based training with RNE Regularization

This folder contains code for RNE for energy-based training on ALDP. 

## 🛠️Env

Our code runs with ```python==3.11``` and ```pytorch==2.1.0```. However, it should be compatible with other version as well.
Additionally, the code requires to install ```openmm``` and ```openmmtools``` as follows:

```
conda install -c conda-forge openmm openmmtools
```

## Data preparation

You may put your own diffusion training data in this folder. We provide two trajectories for ALDP, `trajectory800.h5`, `trajectory300.h5`. These files are available at Huggingface at [https://huggingface.co/datasets/JJHE/RNE-ALDP](https://huggingface.co/datasets/JJHE/RNE-ALDP).

## ✨Training

From this directory (`Applications/energy_based_training`), run energy-based training with RNE regularization

```
python train.py --data trajectory300.h5 --rne_weight 1e3 --n_epoch 200000 --batch_size 64
```

## ✨Evaluation

Diffusion reverse-SDE sampling:

```
python eval.py --mode diffusion --checkpoint ema_net1e3.pt
```

Langevin MCMC on the learned EBM energy:

```
python eval.py --mode mcmc --checkpoint ema_net1e3.pt 
```

