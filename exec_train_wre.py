import os
os.environ['OMP_NUM_THREADS'] = "2"
from torch_geometric.loader import DataLoader
from util.chem import load_elem_attrs
from util.data import load_dataset
from model.util import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import pickle
import sys

torch.set_num_threads(2)

start = time.time()

# Available models: AttFP
model_name = 'AttFP'

task = 'reg'
n_folds = 5

n_epochs = 500
early_stopping = False
patience = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

list_models = list()
if task == 'clf':
    list_acc = list()
    list_f1 = list()
    list_roc_auc = list()
elif task == 'reg':
    list_mae = list()
    list_rmse = list()
    list_r2 = list()
else:
    raise Exception(f'Task {task} is not available.')
dim_out, criterion = (2, torch.nn.CrossEntropyLoss()) if task == 'clf' else (1, torch.nn.MSELoss())
elem_attrs = load_elem_attrs('res/matscholar-embedding.json')



with open('preds/train_result_wre.txt', 'w') as f:
    sys.stdout = f


    dataset_name = 'RE'
    batch_size = 256
    init_lr = 0.0005
    l2_coeff = 1e-6
    dim_hidden = 256
    n_gnn = 4

        
    nrmlz = False

    # Load the dataset.
    dataset, tgt_avg, tgt_std = load_dataset(path_dataset='dataset/{}.xlsx'.format(dataset_name),
                           elem_attrs=elem_attrs,
                           idx_smiles=0,
                           idx_target=1,
                           task=task,
                           pu=False,
                           normalization=nrmlz,
                           calc_pos=False
                           )

    try:
        for k in range(n_folds):
            loader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            model = get_model(model_name=model_name,
                                  dim_node_feat=dataset[0].x.shape[1],
                                  dim_edge_feat=dataset[0].edge_attr.shape[1],
                                  dim_hidden=dim_hidden,
                                  n_gnnlayer=n_gnn,
                                  dim_out=dim_out).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=l2_coeff)
            # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.2, verbose=True)

            # Optimize model parameters.
            for epoch in range(n_epochs):
                loss_train = fit(model, loader_train, optimizer, criterion)

            torch.save(model.state_dict(), 'save/model_{}_{}fold.pt'.format(dataset_name, k))


    finally:
        print('Training complete')

