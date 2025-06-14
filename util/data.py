import pandas
from itertools import chain
from tqdm.auto import tqdm
from util.chem import *
import sys


def load_dataset(path_dataset, elem_attrs, idx_smiles, idx_target, task, p_label=0, pu=False, normalization=False, calc_pos=False):
    data = pandas.read_excel(path_dataset).values.tolist()

    # Target value normalization.
    if normalization:
        tgt_list = []
        for datum in data:
            tgt_list.append(datum[idx_target])
        avg, std = numpy.average(tgt_list), numpy.std(tgt_list)
        for datum in data:
            datum[idx_target] = (datum[idx_target] - avg) / std
    else:
        avg, std = None, None

    dataset = list()

    for i in tqdm(range(len(data)), desc="Loading data", file=sys.stderr):
        mg = get_mol_graph(data[i][idx_smiles], elem_attrs, calc_pos=calc_pos)

        if mg is not None:
            # In PU learning, set positive and unlabeled data's target value as 0 and 1, respectively.
            if pu:
                data[i][idx_target] = p_label if data[i][idx_target] == p_label else int(1 - p_label)
            mg.y = torch.tensor(data[i][idx_target], dtype=torch.long).view(1) if task == 'clf' else (
                torch.tensor(data[i][idx_target], dtype=torch.float).view(1, 1))
            dataset.append(mg)

    return dataset, avg, std




def get_k_folds(dataset, n_folds, random_seed=None):
    if random_seed is not None:
        try:
            numpy.random.seed(random_seed)
        except:
            numpy.random.seed(int(random_seed))

    k_folds = list()
    idx_rand = numpy.array_split(numpy.random.permutation(len(dataset)), n_folds)

    for i in range(0, n_folds):
        idx_train = list(chain.from_iterable(idx_rand[:i] + idx_rand[i + 1:]))
        idx_test = idx_rand[i]
        dataset_train = [dataset[idx] for idx in idx_train]
        dataset_test = [dataset[idx] for idx in idx_test]
        k_folds.append([dataset_train, dataset_test])

    return k_folds
