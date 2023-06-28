from problems import NASBench101, NASBench201

problem_configuration = {
    'NAS101': {
        'dataset': 'CIFAR-10',
    },
    'NAS201-C10': {
        'dataset': 'CIFAR-10',
    },
    'NAS201-C100': {
        'dataset': 'CIFAR-100',
    },
    'NAS201-IN16': {
        'dataset': 'ImageNet16-120',
    }
}

problem_configuration4algorithm = {
    # single-training-based search
    'val_error': {
        '101': {'f0': 'params', 'f1': 'val_error_12'},
        '201': {'f0': 'FLOPs', 'f1': 'val_error_12'},
    },
    'train_loss': {
        '201': {'f0': 'FLOPs', 'f1': 'train_loss_12'}
    },
    'val_loss': {
        '201': {'f0': 'FLOPs', 'f1': 'val_loss_12'}
    },
    'train_error': {
        '101': {'f0': 'params', 'f1': 'train_error_12'},
        '201': {'f0': 'FLOPs', 'f1': 'train_error_12'}
    },
    'MOENAS_PSI': {
        '101': {'f0': 'params', 'f1': 'val_error_12'},
        '201': {'f0': 'FLOPs', 'f1': 'val_error_12'}
    },
    'MOENAS_TF_PSI': {
        '101': {'f0': 'params', 'f1': 'val_error_12'},
        '201': {'f0': 'FLOPs', 'f1': 'val_error_12'}
    },
    'ENAS_TFI': {
        '101': {'f0': 'params', 'f1': 'val_error_12'},
        '201': {'f0': 'FLOPs', 'f1': 'val_error_12'}
    },

    # single-training-free search
    'synflow': {
        '101': {'f0': 'params', 'f1': '-synflow'},
        '201': {'f0': 'FLOPs', 'f1': '-synflow'}
    },
    'jacov': {
        '101': {'f0': 'params', 'f1': '-jacov'},
        '201': {'f0': 'FLOPs', 'f1': '-jacov'}
    },
    'snip': {
        '101': {'f0': 'params', 'f1': '-snip'},
        '201': {'f0': 'FLOPs', 'f1': '-snip'}
    },
    'grasp': {
        '101': {'f0': 'params', 'f1': '-grasp'},
        '201': {'f0': 'FLOPs', 'f1': '-grasp'}
    },
    'grad_norm': {
        '101': {'f0': 'params', 'f1': '-grad_norm'},
        '201': {'f0': 'FLOPs', 'f1': '-grad_norm'}
    },
    'fisher': {
        '101': {'f0': 'params', 'f1': '-fisher'},
        '201': {'f0': 'FLOPs', 'f1': '-fisher'}
    },

    # multi-training-free search
    'E_TF_MOENAS': {
        '101': {'f0': 'params', 'f1': ['-synflow', '-jacov']},
        '201': {'f0': 'FLOPs', 'f1': ['-synflow', '-jacov']}
    },
    'E_TF_MOENAS_C': {
        '101': {'f0': 'params', 'f1': ['-synflow', '-jacov']},
        '201': {'f0': 'FLOPs', 'f1': ['-synflow', '-jacov']}
    },
}

def get_problems(problem_name, **kwargs):
    config = problem_configuration[problem_name]
    if problem_name == 'NAS101':
        return NASBench101(dataset=config['dataset'], max_eval=kwargs['max_eval'], **kwargs)
    elif 'NAS201' in problem_name:
        return NASBench201(dataset=config['dataset'], max_eval=kwargs['max_eval'], **kwargs)
    else:
        raise ValueError(f'Not supporting this problem - {problem_name}.')

def get_algorithm(algorithm_name, **kwargs):
    if algorithm_name in ['val_error', 'val_loss', 'train_loss']:
        from algorithms import NSGAII
        return NSGAII(name=f'MOENAS ({algorithm_name})')

    elif algorithm_name in ['synflow', 'jacov', 'grasp', 'fisher', 'snip', 'grad_norm']:
        from algorithms import TF_NSGAII
        return TF_NSGAII(name=f'TF-MOENAS ({algorithm_name})')

    elif algorithm_name == 'E-TF-MOENAS-C':
        from algorithms import E_TF_NSGAII_C
        return E_TF_NSGAII_C(name=algorithm_name)

    elif algorithm_name == 'E-TF-MOENAS':  # Family of E_TF_MOENAS variants (k >= 2)
        from algorithms import E_TF_NSGAII
        return E_TF_NSGAII(name=algorithm_name)

    elif algorithm_name == 'MOENAS_PSI':
        from algorithms import MOENAS_PSI
        return MOENAS_PSI(name=algorithm_name)

    elif algorithm_name == 'MOENAS_TF_PSI':
        from algorithms import MOENAS_TF_PSI
        return MOENAS_TF_PSI(name=algorithm_name)

    elif algorithm_name == 'ENAS_TFI':
        from algorithms import ENAS_TFI
        return ENAS_TFI(name=algorithm_name, path_pre_pop='./pre_pop')

    else:
        raise ValueError(f'Not supporting this algorithm - {algorithm_name}.')


if __name__ == '__main__':
    pass
