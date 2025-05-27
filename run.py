from GENE import GENE
from Datagenerator import Generator
import argparse
from castle.metrics import MetricsDAG

parser = argparse.ArgumentParser()

# -----------configurations ------

parser.add_argument('--node_num', type=int, default=10,
                    help='number of nodes')
parser.add_argument('--density', type=int, default=2,
                    help='expected density for graph')
parser.add_argument('--sem', type=str, default='mim',
                    choices=['mim', 'mlp', 'gp'],
                    help='nonlinear form')
parser.add_argument('--sample_size', type=int, default=3000,
                    help='the number of samples of data')
parser.add_argument('--data_generator', type=str, default='gcastle',
                    choices=['gcastle', 'cdt', 'sachs'],
                    help='library used to generate data')
parser.add_argument('--seed', type=int, default=1,
                    help='randomseed')
parser.add_argument('--standardize', type=bool, default=False,
                    help='wheter to standardize data')

if __name__ == '__main__':
    args = parser.parse_args()
    data, true_graph = Generator(node_num=args.node_num, density=args.density, sem=args.sem,sample_size=args.sample_size
                                 , data_generator=args.data_generator,seed = args.seed,standardize = args.standardize)
    order, est_matr,ex_t = GENE(train_data=data, test_data=data)
    mt = MetricsDAG(est_matr, true_graph)
    print("F1", mt.metrics['F1'], "SHD", mt.metrics['shd'])
