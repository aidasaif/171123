from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, SubsetRandomSampler
from geochem import *
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


start_time = time.time()

# data preparation
data_js = json.loads(session.query(AnalysisMLP).filter_by(id=12).first().data)
data = pd.DataFrame(data_js)
data['mark'] = data['mark'].replace({'empty': 0, 'bitum': 1})
data['mark'] = data['mark'].replace({'пусто': 0, 'нефть': 1})

input_cols = list(data.columns[2:])
output_cols = ['mark']


def dataframe_to_arrays(data):
    df = data.copy(deep=True)
    input_array = df[input_cols].to_numpy()
    target_array = df[output_cols].to_numpy()
    input_scaler = StandardScaler()
    input_array = input_scaler.fit_transform(input_array)
    return input_array, target_array


inputs_array, targets_array = dataframe_to_arrays(data)
inputs = torch.from_numpy(inputs_array).type(torch.float)
targets = torch.from_numpy(targets_array).type(torch.float)

dataset = TensorDataset(inputs, targets)
train_ds, val_ds = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(10))
print(train_ds, type(train_ds))
batch_size = 50
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size)


def hidden_block(input_dim, output_dim, dropout_rate):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(output_dim),
        torch.nn.Dropout(dropout_rate),
    )

# building a model
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, dropout_rate):
        super(Model, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_units[0])
        self.batch_norm1 = nn.BatchNorm1d(hidden_units[0])
        self.hidden_layers = torch.nn.ModuleList(
            [hidden_block(hidden_units[i], hidden_units[i + 1], dropout_rate) for i in range(len(hidden_units) - 1)]
        )
        self.output_layer = torch.nn.Linear(hidden_units[-1], output_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = self.relu(self.input_layer(x))
        out = self.batch_norm1(out)
        for layer in self.hidden_layers:
            out = self.relu(layer(out))
            out = self.dropout(out)
        out = self.output_layer(out)
        return out


input_dim = len(train_ds[0][0])
output_dim = 1

def fit(epochs, model, optimizer, loss_function, train_dl, val_dl, early_stopping=False):
    best_loss = float('inf')
    patience = 20
    losses = []
    val_losses = []
    all_predictions = []
    all_targets = []
    epoch_val = []

    for epoch in range(epochs):
        epoch_losses = []
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_function(pred, yb)
            l2_lambda = 0.001
            l2_reg = torch.tensor(0.)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())
        losses.append(np.mean(epoch_losses))

        model.eval()
        with torch.no_grad():
            for xb, yb in val_dl:
                predictions = model(xb)
                val_loss = loss_function(predictions, yb)
                all_predictions.append(predictions)
                all_targets.append(yb)
                epoch_val.append(val_loss)
            val_losses.append(np.mean(epoch_val))

        if epoch % 50 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.7f}, Val Loss: {val_losses[-1]:.7f}')

        # early stopping
        if early_stopping:
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 20
            else:
                patience -= 1
                if patience == 0:
                    print(f"     Epoch [{epoch+1}/{epochs}] Early stopping")
                    break
    return all_predictions, all_targets, losses, val_losses

def predictions_to_binary(all_predictions, all_targets, threshold=False):
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    predictions_numpy = all_predictions.numpy()
    targets_numpy = all_targets.numpy()
    fpr, tpr, thresholds = roc_curve(targets_numpy, predictions_numpy)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    if threshold:
        binary_predictions = (predictions_numpy >= optimal_threshold).astype(int)
    else:
        binary_predictions = (predictions_numpy >= 0.5).astype(int)
    return binary_predictions, targets_numpy

def cross_validation(dataset, epochs, model, optimizer, loss_function):
    print(f'----------------- CROSS VALIVATION -----------------\n')
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    accuracy_list = []

    loss_list = []
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)

        train_loader = DataLoader(dataset, batch_size=99, sampler=train_subsampler)
        valid_loader = DataLoader(dataset, batch_size=99, sampler=valid_subsampler)

        all_predictions, all_targets, losses, val_losses = fit(epochs, model, optimizer, loss_function,
                                                               train_loader, valid_loader)
        loss_list.append(val_losses)

        predictions, targets = predictions_to_binary(all_predictions, all_targets, threshold=True)
        accuracy = accuracy_score(targets, predictions)
        accuracy_list.append(accuracy)
        print(f'--- Accuracy: {accuracy:.4f} ---')

    flattened_list = [item for sublist in loss_list for item in sublist]
    print('Training Ended')
    print('Average Loss: {}\n'.format(np.mean(flattened_list)))

    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(1, len(accuracy_list) + 1), accuracy_list, color='skyblue')
    for bar, value in zip(bars, accuracy_list):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=10)
    plt.title('График кросс-валидации')
    plt.xlabel('Номер фолда')
    plt.ylabel('Accuracy')
    plt.show()

epochs = 150
best_learning_rate = 0.0036825504095464845
best_num_hidden_units = [47, 53, 59, 68, 74, 38]
best_dropout_rate = 0.20848781781365544
best_weight_decay = 0.001789698835578157

model = Model(input_dim, output_dim, best_num_hidden_units, best_dropout_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)
loss_function = torch.nn.BCEWithLogitsLoss()

# Кросс валидация
cross_validation(dataset, epochs, model, optimizer, loss_function)

print("Время выполнения программы:", time.time() - start_time, "секунд")