import pickle as p

def do_each_gen(**kwargs):
    algorithm = kwargs['algorithm']

    algorithm.nEvals_runningtime_each_gen.append([algorithm.n_eval, algorithm.running_time_history[-1]])
    algorithm.E_Archive_search_each_gen.append(algorithm.E_Archive_search_history[-1].copy())

def finalize(**kwargs):
    algorithm = kwargs['algorithm']

    p.dump(algorithm.nEvals_runningtime_each_gen, open(f'{algorithm.res_path}/#Evals_runningtime_each_gen.p', 'wb'))
    p.dump(algorithm.E_Archive_search_each_gen, open(f'{algorithm.res_path}/E_Archive_search_each_gen.p', 'wb'))

    p.dump([algorithm.nEvals_history, algorithm.E_Archive_search_history],
           open(f'{algorithm.res_path}/#Evals_and_Elitist_Archive_search.p', 'wb'))