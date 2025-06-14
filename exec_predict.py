import numpy
from pandas import DataFrame
from sklearn.metrics import r2_score, mean_absolute_error
from torch_geometric.loader import DataLoader
from util.chem import load_elem_attrs
from util.data import load_dataset, get_k_folds
from model.util import *
import pickle


# Experiment settings.
dataset_name = 'initial_dataset'
target_list = ['logGWP', 'logtau', 'RE', 'LFL', 'Tc']
task = 'reg'
n_folds = 5
model_name = 'AttFP'
list_r2 = list()
list_mae = list()
list_preds = list()
list_targets = list()
dim_out = 2 if task == 'clf' else 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prediction_list = []
std_list = []

hpbs = [64, 64, 256, 64, 64]
hplr= [0.001, 0.001, 0.0005, 0.001, 0.0005]
hpl2 = [5e-6, 1e-6, 1e-6, 5e-6, 5e-6]
hpdim = [64, 128, 256, 128, 64]
hpgnn = [2, 3, 4, 3, 2]

#
elem_attrs = load_elem_attrs('res/matscholar-embedding.json')
for i, tgt_name in enumerate(target_list):
    pred_stat = []

    batch_size = hpbs[i]
    dim_hidden = hpdim[i]
    n_gnn = hpgnn[i]

    if tgt_name == 'Tc':
        nrmlz = True
    else:
        nrmlz = False

    # Load the dataset.

    dataset, tmp_avg, tmp_std = load_dataset(path_dataset='dataset/{}.xlsx'.format(dataset_name),
                           elem_attrs=elem_attrs,
                           idx_smiles=0,
                           idx_target=1,
                           task=task,
                           pu=False,
                           normalization=nrmlz,
                           calc_pos=False)
    k_folds = get_k_folds(dataset, n_folds=n_folds, random_seed=None)

    smls_list = [dataset[i].smls for i in range(len(dataset))]

    for k in range(0, n_folds):
        loader_test = DataLoader(dataset, batch_size=batch_size)

        # Load a trained model.
        model = get_model(model_name=model_name,
                          dim_node_feat=dataset[0].x.shape[1],
                          dim_edge_feat=dataset[0].edge_attr.shape[1],
                          dim_hidden=dim_hidden,
                          n_gnnlayer=n_gnn,
                          dim_out=dim_out).to(device)
        model.load_state_dict(torch.load('save/model_{}_{}fold.pt'.format(tgt_name, k), map_location=device))

        # Evaluate the trained prediction model on the test dataset.
        preds_test, targets_test = test(model, loader_test)
        # if k == 0: preds = torch.zeros(preds_test.shape)
        pred_stat.append(preds_test.numpy())
        # Save the prediction results.

    pred_np = numpy.mean(numpy.array(pred_stat), axis=0)
    pred_std = numpy.std(numpy.array(pred_stat), axis=0)

    if nrmlz:
        with open("./util/tcavgstd.pkl", "rb") as fr:
            tgt_avg, tgt_std = pickle.load(fr)
        pred_np = tgt_std * pred_np + tgt_avg
        targets_test = tgt_std * targets_test + tgt_avg
        pred_std *= tgt_std

    if task == 'clf':
        preds_clf = numpy.argmax(pred_np, axis=1)
        prediction_list.append(preds_clf)
    else:
        prediction_list.append(pred_np)
    std_list.append(pred_std)

pred_results = numpy.column_stack([numpy.array(smls_list).reshape(-1, 1), prediction_list[0], prediction_list[1],
                                   prediction_list[2], prediction_list[3], prediction_list[4]])
std_results = numpy.column_stack([numpy.array(smls_list).reshape(-1, 1), std_list[0], std_list[1],
                                   std_list[2], std_list[3], std_list[4]])
DataFrame(pred_results).to_excel(f'preds/supervised_result.xlsx', header=['SMILES'] + target_list, index=False)
DataFrame(std_results).to_excel(f'preds/supervised_uncertainty.xlsx', header=['SMILES'] + target_list, index=False)


