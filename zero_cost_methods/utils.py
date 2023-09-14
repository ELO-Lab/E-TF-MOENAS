from .predictors import ZeroCostV2
from utils.utils import X2matrices
from collections import namedtuple

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

def get_config_for_zero_cost_predictor(problem, database_path, seed):
    config = {
        'search_space': problem.name,
        'dataset': problem.dataset,
        'root_data': database_path + '/dataset',
        'search': {
                'batch_size': 256,
                'data_size': 25000,
                'cutout': False,
                'cutout_length': 16,
                'cutout_prob': 1.0,
                'train_portion': 0.7,
                'seed': seed
        }
    }
    return config

def get_zero_cost_predictor(config, method_type):
    if method_type == 'grad_norm':
        predictor = ZeroCostV2(config, batch_size=64, method_type='grad_norm')
    elif method_type == 'grasp':
        predictor = ZeroCostV2(config, batch_size=64, method_type='grasp')
    elif method_type == 'jacov':
        predictor = ZeroCostV2(config, batch_size=64, method_type='jacov')
    elif method_type == 'snip':
        predictor = ZeroCostV2(config, batch_size=64, method_type='snip')
    elif method_type == 'fisher':
        predictor = ZeroCostV2(config, batch_size=64, method_type='fisher')
    elif method_type == 'synflow':
        predictor = ZeroCostV2(config, batch_size=64, method_type='synflow')
    elif method_type == 'synflow-jacov':
        method_type_lst = ['synflow', 'jacov']
        predictor = ZeroCostV2(config, batch_size=64, method_type=method_type_lst)
    else:
        if isinstance(method_type, list):
            supported_list = ['grad_norm', 'grasp', 'jacov', 'snip', 'fisher', 'synflow']
            for method in method_type:
                if method not in supported_list:
                    raise ValueError(f'Just supported "grad_norm"; "grasp"; "jacob"; "snip"; and "synflow", not {method_type}.')
            predictor = ZeroCostV2(config, batch_size=64, method_type=method_type)
        else:
            raise ValueError(
                f'Just supported "grad_norm"; "grasp"; "jacob"; "snip"; and "synflow", not {method_type}.')
    # predictor.pre_process()
    return predictor

def modify_input_for_fitting(X, problem_name):
    if problem_name == 'NASBench101':
        edges_matrix, ops_matrix = X2matrices(X)
        X_modified = {'matrix': edges_matrix, 'ops': ops_matrix}
    elif problem_name == 'NASBench201':
        OPS_LIST = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        X_modified = f'|{OPS_LIST[X[0]]}~0|+' \
                     f'|{OPS_LIST[X[1]]}~0|{OPS_LIST[X[2]]}~1|+' \
                     f'|{OPS_LIST[X[3]]}~0|{OPS_LIST[X[4]]}~1|{OPS_LIST[X[5]]}~2|'
    elif problem_name == 'NASBench301':
        """
        Convert the compact representation to a genotype
        """
        OPS = [
            "max_pool_3x3",
            "avg_pool_3x3",
            "skip_connect",
            "sep_conv_3x3",
            "sep_conv_5x5",
            "dil_conv_3x3",
            "dil_conv_5x5",
        ]
        genotype_0 = []
        for i in range(0, 16, 2):
            genotype_0.append((OPS[X[i]], X[i + 1]))
        genotype_1 = []
        for i in range(16, 32, 2):
            genotype_1.append((OPS[X[i]], X[i + 1]))

        X_modified = Genotype(
            normal=genotype_0,
            normal_concat=[2, 3, 4, 5],
            reduce=genotype_1,
            reduce_concat=[2, 3, 4, 5],
        )
    else:
        raise ValueError(f'Not support this problem: {problem_name}.')
    return X_modified