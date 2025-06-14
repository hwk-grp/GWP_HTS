import torch
from model.attfp import AttentiveFP

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(model_name, dim_node_feat, dim_edge_feat, dim_hidden, n_gnnlayer, dim_out):
    if model_name == 'AttFP':
        return AttentiveFP(in_channels=dim_node_feat, hidden_channels=dim_hidden, out_channels=dim_out,
                           edge_dim=dim_edge_feat, num_layers=n_gnnlayer, num_timesteps=2)
    else:
        raise EOFError


def fit(model, data_loader, optimizer, criterion):
    train_loss = 0

    model.train()
    for batch in data_loader:
        batch = batch.to(device)
        preds = model(batch)

        loss = criterion(preds, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(data_loader)


def loss_return(model, data_loader, criterion):
    valid_loss = 0

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            preds = model(batch)

            loss = criterion(preds, batch.y)
            valid_loss += loss.item()

    return valid_loss / len(data_loader)


def test(model, data_loader):
    list_preds = list()
    list_targets = list()

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            preds = model(batch)
            list_preds.append(preds)
            list_targets.append(batch.y)
        #     print(preds.size(), batch.y.size())
        # print(list_preds)
        # print(list_targets)
    return torch.cat(list_preds, dim=0).cpu(), torch.cat(list_targets, dim=0).cpu()
