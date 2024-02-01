from argparse import ArgumentParser

from helpers.dataset import DataSet
from helpers.model_type import ModelType
from helpers.classes import Pool


def parse_arguments():
    parser = ArgumentParser()
    # dataset
    parser.add_argument("--dataset", dest="dataset", default=DataSet.ER, type=DataSet.from_string,
                        choices=list(DataSet), required=False)
    parser.add_argument('--graph_size', dest='graph_size', type=int, required=True)
    parser.add_argument("--num_graph_samples", dest="num_graph_samples", default=100, type=int, required=False)
    parser.add_argument('--rw_pos_length', dest='rw_pos_length', type=int, default=0, required=False)

    # model
    parser.add_argument("--model_type", dest="model_type", default=ModelType.MEAN_GNN,
                        type=ModelType.from_string, choices=list(ModelType), required=False)
    parser.add_argument("--num_layers", dest="num_layers", default=3, type=int, required=False)
    parser.add_argument("--in_dim", dest="in_dim", default=128, type=int, required=False)
    parser.add_argument("--hidden_dim", dest="hidden_dim", default=128, type=int, required=False)
    parser.add_argument("--out_dim", dest="out_dim", default=5, type=int, required=False)
    parser.add_argument("--pool", dest="pool", default=Pool.MEAN, type=Pool.from_string,
                        choices=list(Pool), required=False)

    # reproduce
    parser.add_argument("--seed", dest="seed", type=int, default=0, required=False)
    parser.add_argument('--gpu', dest="gpu", type=int, required=False)
    return parser.parse_args()
