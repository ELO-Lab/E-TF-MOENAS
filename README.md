# E-TF-MOENAS: Enhanced Training-free Multi-Objective Neural Architecture Search
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Ngoc Hoang Luong, Quan Minh Phan, An Vo, Tan Ngoc Pham, Dzung Tri Bui

## Setup
- Clone repo.
- Install necessary packages.
```
$ cd E-TF-MOENAS
$ bash install.sh
```
In our experiments, we do not use directly the API benchmarks published in their repos (i.e., [NAS-Bench-101](https://arxiv.org/abs/1902.09635), [NAS-Bench-201](https://arxiv.org/abs/2001.00326)).
Instead, we access their databases and only logging necessary information to create smaller-size databases.
 
You can compare our databases and the original databases in [check_log_database.ipynb](check_log_database.ipynb)
## Reproducing the results
You can reproduce the results in our article by running the below script:
```shell
$ python main.py --problem [NAS101, NAS201-C10] --algorithm <search-strategy> 
```
All problems in our article are bi-objective minimization problems.
The objectives consist of the test error rate and one of the complexity/efficiency metrics (e.g., FLOPs, #params, latency).

To simulate the real-world NAS settings (the test performance can not be accessed), the search strategies have to optimize another metric instead of the test error rate during the search.
Here are the list of supported optimizers and their used performance metrics during the search process:

| Search Strategy                   |Description            |  NAS-Bench-101                         | NAS-Bench-201                          |            
|:--------------------------|:----------------------|:---------------------------------------|:---------------------------------------|
|`val_error`          |Optimizer validation error (epoch 12th) | :heavy_check_mark: | :heavy_check_mark: |
|`val_loss`           |Optimizer validation loss (epoch 12th) | :x: | :heavy_check_mark: |
|`train_loss`         |Optimizer [training loss](https://arxiv.org/abs/2006.04492) (epoch 12th) | :x: | :heavy_check_mark: |
|`synflow`         |Optimizer [Synaptic Flow](https://arxiv.org/abs/2006.05467) metric | :heavy_check_mark: | :heavy_check_mark: |
|`jacov`         |Optimizer [Jacobian Covariance](https://arxiv.org/abs/2006.04647v1) metric | :heavy_check_mark: | :heavy_check_mark: |
|`snip`         |Optimizer [SNIP](https://arxiv.org/abs/1810.02340) metric | :heavy_check_mark: | :heavy_check_mark: |
|`grad_norm`         |Optimizer [Grad Norm](https://arxiv.org/abs/2101.08134) metric | :heavy_check_mark: | :heavy_check_mark: |
|`grasp`         |Optimizer [GRASP](https://arxiv.org/abs/2002.07376) metric  | :heavy_check_mark: | :heavy_check_mark: |
|`fisher`         |Optimizer [Fisher](https://arxiv.org/abs/1906.04113) metric  | :heavy_check_mark: | :heavy_check_mark: |
|`E-TF-MOENAS`        |Optimizer [Synaptic Flow](https://arxiv.org/abs/2006.05467) and [Jacobian Covariance](https://arxiv.org/abs/2006.04647v1) | :heavy_check_mark: | :heavy_check_mark: |
|`E-TF-MOENAS-C`        |Optimizer the sum of [Synaptic Flow](https://arxiv.org/abs/2006.05467) and [Jacobian Covariance](https://arxiv.org/abs/2006.04647v1) | :heavy_check_mark: | :heavy_check_mark: |
|`Free_MOENAS`        |Optimizer the sum of three training-free metrics reported in [FreeREA](https://arxiv.org/abs/2207.05135)  | :heavy_check_mark: | :heavy_check_mark: |
|[`MOENAS_PSI`](https://github.com/ELO-Lab/MOENAS-TF-PSI)         |Similar to `val_error` but perform Pareto Local Search at each generation | :heavy_check_mark: | :heavy_check_mark: |
|[`MOENAS_TF_PSI`](https://github.com/ELO-Lab/MOENAS-TF-PSI)         |Similar to `MOENAS_PSI` but perform training-free Pareto Local Search | :heavy_check_mark: | :heavy_check_mark: |
|[`ENAS_TFI`](https://github.com/ELO-Lab/ENAS-TFI)         |Similar to `val_error` but perform a training-free warm-up stage at the beginning of the search | :heavy_check_mark: | :heavy_check_mark: |

Note #1: All variants use NSGA-II as the search optimizer.

Note #2: In our article, we only report the performance of algorithms obtained on CIFAR-10. But you also search on CIFAR-100 and ImageNet16-120 (NAS-Bench-201) by changing value of `--problem` hyperparameter . 
## Search with different hyperparameters
Moreover, you can search with different hyperparameter settings.
### For environment
- `problem`: NAS201-C100, NAS201-IN16
- `f0`: the complexity/efficiency objective (flops, params, latency (latency is only supported in NAS-Bench-201))
- `n_run`: the number times of running algorithms (default: `31`)
- `max_eval`: the maximum number of evaluation each run (default: `3000`)
- `init_seed`: the initial random seed (default: `0`)
- `res_path`: the path for logging results (default: `./exp_res`)
- `debug`: print the search performance at each generation if `debug` is `True` (default: `False`)
### For algorithms
- `pop_size`: the population size


## Transferability Evaluation (for NAS-Bench-201 only)
We evaluate the transferability of algorithms by evaluating the found architectures (search on CIFAR-10) on CIFAR-100 and ImageNet16-120 in our article.

Source code for transferability evaluation can be found [here](transferability_evaluation.ipynb).

## Visualization and T-test
Source code for results visualization can be found [here](visualization101.ipynb) (for NAS-Bench-101) and [here](visualization201.ipynb) (for NAS-Bench-201).

## Acknowledgement
Our source code is inspired by:
- [pymoo: Multi-objective Optimization in Python](https://github.com/anyoptimization/pymoo)
- [NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm](https://github.com/ianwhale/nsga-net)