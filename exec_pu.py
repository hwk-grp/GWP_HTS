import numpy
import pandas
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.loader import DataLoader
from util.chem import load_elem_attrs
from util.data import load_dataset
from model.util import *
import time
import random
from collections import defaultdict


start = time.time()

# Available models: AttFP,
model_name = 'AttFP'

# Set the positive data's label as 0 or 1
positive_label = 0

dataset_name = 'inhalation_for_pu0'

# Number of iterations
n_bag = 100

# Task is fixed to classification (clf)
task = 'clf'
batch_size = 64
init_lr = 0.001
l2_coeff = 5e-6
n_epochs = 200
dim_hidden = 128
n_gnn = 3
dim_out, criterion = 2, torch.nn.CrossEntropyLoss()
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the dataset.
elem_attrs = load_elem_attrs('res/matscholar-embedding.json')
dataset, avg, std = load_dataset(path_dataset='dataset/{}.xlsx'.format(dataset_name),
                       elem_attrs=elem_attrs,
                       idx_smiles=0,
                       idx_target=1,
                       task=task,
                       p_label=positive_label,
                       pu=True,
                       calc_pos=False
                       )
random.seed(0)
random.shuffle(dataset)

# Divide positive sample and unlabeled sample
pos = [dataset[i] for i in range(len(dataset)) if dataset[i].y == positive_label]
unk = [dataset[i] for i in range(len(dataset)) if dataset[i].y != positive_label]
all_indices = set(range(len(unk)))

# Data collection
coll_data = []
for bag in range(n_bag):
    # guarantee randomness for each {bag}
    random.seed(bag)

    # Label some portion of unlabeled data as a negative sample
    neg = random.sample(unk, len(pos))

    # merge positive and (assumed) negative data
    train_tot = pos + neg

    # Unlabeled data (w/o negative data) becomes test data
    unlabel = [item for item in unk if item not in neg]

    # print(len(train_tot), len(unlabel))
    coll_data.append((train_tot, unlabel))
    if bag % 10 == 9: print(f'Data partition: bagging {bag + 1}/{n_bag}')

tot_dict_list = []

for bag in range(n_bag):
    loader_train = DataLoader(coll_data[bag][0], batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(coll_data[bag][1], batch_size=batch_size)
    model = get_model(model_name=model_name,
                      dim_node_feat=dataset[0].x.shape[1],
                      dim_edge_feat=dataset[0].edge_attr.shape[1],
                      dim_hidden=dim_hidden,
                      n_gnnlayer=n_gnn,
                      dim_out=dim_out).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=l2_coeff)

    for epoch in range(n_epochs):
        loss_train = fit(model, loader_train, optimizer, criterion)

    preds_test, targets_test = test(model, loader_test)

    # Save result for each {bag}
    smiles_test = []
    labeling_dict = {}
    for i in range(len(coll_data[bag][1])):
        smiles_test.append(coll_data[bag][1][i].smls)
    if task == 'clf':
        max_preds_test = numpy.argmax(preds_test.numpy(), axis=1)
        for i in range(len(smiles_test)):
            labeling_dict[smiles_test[i]] = max_preds_test[i]
    tot_dict_list.append(labeling_dict)
    if bag % 5 == 4: print(f'Training: bagging {bag + 1}/{n_bag}')

# Calculate the average score
sums = defaultdict(int)
counts = defaultdict(int)
for d in tot_dict_list:
    for key, value in d.items():
        sums[key] += value
        counts[key] += 1

averaged_dict = {key: sums[key] / counts[key] for key in sums}
df = pandas.DataFrame(list(averaged_dict.items()), columns=['SMILES', 'Toxic'])
df.to_excel(f'preds/pu_result.xlsx', index=False)
