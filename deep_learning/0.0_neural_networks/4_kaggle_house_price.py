'''
Kaggle House Price Prediction
'''


import numpy as np
import pandas as pd
import torch
from torch import nn

from dl_utils.data.downloads import download
from dl_utils.data.loading import load_array
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.plot.figures import plot

_output_dir = infer_project_root() / 'output' / 'kaggle_house'
reset_dir(str(_output_dir))

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net

def log_rmse(net, features, labels):
    # To further stabilize the value when taking the logarithm, set values less than 1 to 1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels, 
          num_epochs, lr, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    # The Adam optimizer is used here
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, lr, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, lr, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                          xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs], 
                          legend=['train', 'valid'])
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

def train_and_pred(train_features, test_features, train_labels, test_data, 
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None, 
                        num_epochs, lr, weight_decay, batch_size)
    plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
                  ylabel='rmse', xlim=[1, num_epochs], ylim=[0, 1])
    print(f'train log rmse {float(train_ls[-1]):f}')
    # Apply the network to the test set
    net.eval()
    with torch.no_grad():
        preds = net(test_features).cpu().numpy()
    # Reformat it for export to Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv(_output_dir / 'submission.csv', index=False)

if __name__ == '__main__':
    train_data = pd.read_csv(download('kaggle_house_train'))
    test_data = pd.read_csv(download('kaggle_house_test'))
    # remove the Id column (first column) from both train and test data
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

    # If test data is unavailable, the mean and standard deviation can be calculated from the training data
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())) # Standardize the data
    # After standardizing the data, all means disappear, so we can set the missing values to 0
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    # dummy_na=True treats na (missing values) as valid feature values and creates indicator features for them
    all_features = pd.get_dummies(all_features, dummy_na=True) # One-hot encoding for categorical features
    all_features = all_features.astype(np.float32)
    print(all_features.shape)

    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

    # Train a linear regression model for baseline performance
    loss = nn.MSELoss()
    in_features = train_features.shape[1]

    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64  
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, 
                              weight_decay, batch_size)

    # plt.ioff()
    # plt.show()

    print(f'{k}-fold validation: avg train log rmse {float(train_l):f}, '
          f'avg valid log rmse {float(valid_l):f}')

    train_and_pred(train_features, test_features, train_labels, test_data, 
                   num_epochs, lr, weight_decay, batch_size)
    
    # plt.ioff()
    # plt.show()
