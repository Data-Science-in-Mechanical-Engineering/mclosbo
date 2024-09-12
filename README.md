# MCLoSBO

## Lipschitz Safe Bayesian Optimization for Automotive Control

This repository contains code for the paper ["Lipschitz Safe Bayesian Optimization for Automotive Control"](https://arxiv.org/abs/2312.08058) which has been accepted to the [2024 Conference for Decision and Control](https://cdc2024.ieeecss.org/) in Milano, Italy.

If you find our code or paper useful, please consider citing
```
@article{menn2024lipschitz,
  title={Lipschitz Safe {Bayesian} Optimization for Automotive Control},
  author={Menn, Johanna and Pelizzari, Pietro and Fleps-Dezasse, Michael and Trimpe, Sebastian},
  booktitle={2024 63nd IEEE Conference on Decision and Control (CDC)},
  year={2024},
  organization={IEEE}
}
```
---

We propose a new algorithm **MCLoSBO**, that uses the safety mechanism of [Lipschitz-only Safe Bayesian Optimization (LoSBO)](https://openreview.net/forum?id=tgFHZMsl1N&noteId=7pMQ8SJ4Mz) in a multiple constraints setting.

Our implementation uses [SafeOptMC](https://github.com/befelix/SafeOpt) as a basis and a baseline for the experiments. 

## Reproduce the experiments

The code in this repository can be used to reproduce the figures and results of our paper. You can reproduce the experiments by runnig the 
```
python experiments.py
```
and reproduce the plot from the paper with
```
python evaluate_simulation_results.py
```

As this project was a collaboration with industry, we cannot provide the orginal simulator, that was used for the experiments in the paper.
As a work around, we queried the simulator with a fine grid over the parameter space and evaluated the objective and constraint functions. 
The results of this gridding are used when running the experiments script. 

## Pip

Into an environment with Python 3.11 you can install the needed packages with
```
pip install -r requirements.txt
```
