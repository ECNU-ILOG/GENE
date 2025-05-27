from castle.datasets import IIDSimulation, DAG
import random
import numpy as np
from cdt.data import load_dataset

def Generator(node_num=10, density=2, sem='mlp',sample_size=3000, data_generator='gcastle',seed = 42,standardize = False):
    def standardization(X):
        standardized_X = X.copy()
        for i in range(X.shape[1]):
            column = X[:, i]
            mean = np.mean(column)
            std = np.std(column)
            # print(i, mean, std)
            column = (column - mean)/std
            standardized_X[:, i] = column
        return standardized_X
    if data_generator == 'gcastle':
        weighted_random_dag = DAG.erdos_renyi(n_nodes=node_num, n_edges=node_num*density, weight_range=(0.5, 2.0), seed=seed)
        dataset = IIDSimulation(W=weighted_random_dag, n=sample_size, method='nonlinear', sem_type=sem, noise_scale=1)
        true_graph = dataset.B
        data = dataset.X
    elif data_generator == 'sachs':
        node_num = 11
        sample_size = 853
        samples, _ = load_dataset("sachs")
        list_nmae = samples.columns.values.tolist()
        graph = [('PKC', 'praf'), ('PKC', 'pmek'), ('PKC', 'pjnk'), ('PKC', 'P38'), ('PKC', 'PKA'), ('PKA', 'praf'),
                 ('PKA', 'p44/42'), ('PKA', 'pjnk'), ('PKA', 'P38'), ('praf', 'pmek'), ('PKA', 'pakts473'),
                 ('pmek', 'p44/42'), ('plcg', 'PIP2'), ('plcg', 'PIP3'), ('PIP3', 'PIP2'), ('p44/42', 'pakts473')]
        data = np.array(samples)[:sample_size, :]
        true_graph = np.zeros((node_num, node_num))
        for edge in graph:
            true_graph[list_nmae.index(edge[0])][list_nmae.index(edge[1])] = 1
    if standardize:
        data = standardization(data)
    return data,true_graph