# Inference-time tempering (annealing) with RNE

This folder contains code for RNE on inference-time tempering for ALDP. 

## 🛠️Env

Our code runs with ```python==3.11``` and ```pytorch==2.1.0```. However, it should be compatible with other version as well.
Additionally, the code requires to install ```openmm``` and ```openmmtools``` as follows:

```
conda install -c conda-forge openmm openmmtools
```



## ✨Sampling 

Tempering with RNE (default: $\lambda_{\mathrm{fwd}}0, \lambda_{\mathrm{bwd}}=1$, with reference):

```
python aldp_anneal_mean.py --fwd_lamb 0 --bwd_lamb 1 --fkc 0 --heuristic 0 --ref 1 --bsz 5000 --repeats 50
```

Baselines (at most one of `--fkc` / `--heuristic` may be set):

```
python aldp_anneal_mean.py  --fkc 1 --heuristic 0 --ref 0 --bsz 5000 --repeats 50
python aldp_anneal_mean.py  --fkc 0 --heuristic 1 --ref 0 --bsz 5000 --repeats 50
```

Require the following files in this directory: `ema_net_800k.pt`, `trajectory800.h5`, `trajectory300.h5`. These files are available at Huggingface at [https://huggingface.co/datasets/JJHE/RNE-ALDP](https://huggingface.co/datasets/JJHE/RNE-ALDP).

