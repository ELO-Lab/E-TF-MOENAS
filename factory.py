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
        '101': {'f1': 'val_error_12'},
        '201': {'f1': 'val_error_12'},
    },
    'train_loss': {
        '201': {'f1': 'train_loss_12'}
    },
    'val_loss': {
        '201': {'f1': 'val_loss_12'}
    },
    'train_error': {
        '101': {'f1': 'train_error_12'},
        '201': {'f1': 'train_error_12'}
    },
    'MOENAS_PSI': {
        '101': {'f1': 'val_error_12'},
        '201': {'f1': 'val_error_12'}
    },
    'MOENAS_TF_PSI': {
        '101': {'f1': 'val_error_12'},
        '201': {'f1': 'val_error_12'}
    },
    'ENAS_TFI': {
        '101': {'f1': 'val_error_12'},
        '201': {'f1': 'val_error_12'}
    },
    'Free_EMONAS': {
        '101': {'f1': 'logsynflow + nwot + skip'},
        '201': {'f1': 'logsynflow + nwot + skip'}
    },

    # single-training-free search
    'synflow': {
        '101': {'f1': '-synflow'},
        '201': {'f1': '-synflow'}
    },
    'jacov': {
        '101': {'f1': '-jacov'},
        '201': {'f1': '-jacov'}
    },
    'snip': {
        '101': {'f1': '-snip'},
        '201': {'f1': '-snip'}
    },
    'grasp': {
        '101': {'f1': '-grasp'},
        '201': {'f1': '-grasp'}
    },
    'grad_norm': {
        '101': {'f1': '-grad_norm'},
        '201': {'f1': '-grad_norm'}
    },
    'fisher': {
        '101': {'f1': '-fisher'},
        '201': {'f1': '-fisher'}
    },

    # multi-training-free search
    'E-TF-MOENAS': {
        '101': {'f1': ['-synflow', '-jacov']},
        '201': {'f1': ['-synflow', '-jacov']}
    },
    'E-TF-MOENAS-C': {
        '101': {'f1': ['-synflow', '-jacov']},
        '201': {'f1': ['-synflow', '-jacov']}
    },
}

def get_problems(problem_name, max_eval, **kwargs):
    config = problem_configuration[problem_name]
    if problem_name == 'NAS101':
        return NASBench101(dataset=config['dataset'], max_eval=max_eval, **kwargs)
    elif 'NAS201' in problem_name:
        return NASBench201(dataset=config['dataset'], max_eval=max_eval, **kwargs)
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

    elif algorithm_name == 'Free_EMONAS':
        from algorithms import Free_NSGAII
        return Free_NSGAII(name=algorithm_name)
    else:
        raise ValueError(f'Not supporting this algorithm - {algorithm_name}.')


if __name__ == '__main__':
    pass
