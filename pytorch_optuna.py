import numpy as np
from torch import optim
from func import *
from geochem import *
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import argparse
from filelock import FileLock
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly
import plotly.io as pio

# data preparation
data_js = json.loads(session.query(AnalysisMLP).filter_by(id=2).first().data)
data = pd.DataFrame(data_js)
data['mark'] = data['mark'].replace({'empty': 0, 'bitum': 1})
data['mark'] = data['mark'].replace({'пусто': 0, 'нефть': 1})

input_cols = list(data.columns[2:])
output_cols = ['mark']


def dataframe_to_arrays(data):
    df = data.copy(deep=True)
    input_array = df[input_cols].to_numpy()
    target_array = df[output_cols].to_numpy()
    return input_array, target_array


inputs_array, targets_array = dataframe_to_arrays(data)
inputs = torch.from_numpy(inputs_array).type(torch.float)
targets = torch.from_numpy(targets_array).type(torch.float)

dataset = TensorDataset(inputs, targets)
train_ds, val_ds = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(10))
batch_size = 20
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size)


# building a model
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, dropout_rate):
        super(Model, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_units[0])
        self.batch_norm1 = nn.BatchNorm1d(hidden_units[0]) ####
        self.hidden_layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)]
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


def fit(learning_rate, num_hidden_units, dropout_rate, weight_decay, n_splits=5, patience=10):
    model = Model(input_dim, output_dim, num_hidden_units, dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = torch.nn.BCEWithLogitsLoss()
    epochs = 100

    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for xb, yb in train_dl:
            pred = model(xb)
            pred = torch.sigmoid(pred)  # Применяем сигмоидную функцию активации к предсказанным значениям
            loss = loss_function(pred, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())
        epoch_loss = sum(epoch_losses) / len(epoch_losses)  # Вычисляем среднюю потерю для текущей эпохи
        losses.append(epoch_loss)

    model.eval()
    # Список для хранения всех прогнозов
    all_predictions = []
    all_targets = []
    # Проходим по DataLoader и делаем прогнозы
    with torch.no_grad():
        for xb, yb in val_dl:
            # Получаем прогнозы от модели
            predictions = model(xb)
            # Добавляем прогнозы в общий список
            all_predictions.append(predictions)
            all_targets.append(yb)
    # Объединяем все прогнозы в один тензор
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    # Преобразуем тензор прогнозов в массив NumPy, если нужно
    predictions_numpy = all_predictions.numpy()
    targets_numpy = all_targets.numpy()
    binary_predictions = (predictions_numpy >= 0.5).astype(int)
    accuracy = accuracy_score(targets_numpy, binary_predictions)
    return accuracy


def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
    num_hidden_units = [trial.suggest_int(f'num_hidden_units_layer{i}', 20, 350) for i in range(1, 5)]
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.9)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)

    log_loss = fit(learning_rate, num_hidden_units, dropout_rate, weight_decay)

    return log_loss


optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

print("Number of finished trials:", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

best_learning_rate = trial.params['learning_rate']
best_num_hidden_units = [trial.params[f'num_hidden_units_layer{i}'] for i in range(1, 5)]
best_dropout_rate = trial.params['dropout_rate']
best_weight_decay = trial.params['weight_decay']

final_model = Model(input_dim, output_dim, best_num_hidden_units, best_dropout_rate)
optimizer = torch.optim.Adam(final_model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)
loss_function = torch.nn.BCEWithLogitsLoss()

epoch = 100
best_loss = float('inf')
patience = 15
losses = []
val_losses = []

for epoch in range(epoch):
    epoch_losses = []
    for xb, yb in train_dl:
        pred = final_model(xb)
        pred = torch.sigmoid(pred)
        loss = loss_function(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_losses.append(loss.item())
    losses.append(np.mean(epoch_losses))

    final_model.eval()
    # Список для хранения всех прогнозов
    all_predictions = []
    all_targets = []
    epoch_val = []
    # Проходим по DataLoader и делаем прогнозы
    with torch.no_grad():
        for xb, yb in val_dl:
            # Получаем прогнозы от модели
            predictions = final_model(xb)
            val_loss = loss_function(predictions, yb)
            # Добавляем прогнозы в общий список
            all_predictions.append(predictions)
            all_targets.append(yb)
            epoch_val.append(val_loss)
        val_losses.append(np.mean(epoch_val))

    # early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        patience = 15
    else:
        patience -= 1
        if patience == 0:
            break

# Объединяем все прогнозы в один тензор
all_predictions = torch.cat(all_predictions, dim=0)
all_targets = torch.cat(all_targets, dim=0)
# Преобразуем тензор прогнозов в массив NumPy, если нужно
predictions_numpy = all_predictions.numpy()
targets_numpy = all_targets.numpy()
binary_predictions = (predictions_numpy >= 0.5).astype(int)

matches = 0
for i in range(len(binary_predictions)):
    if binary_predictions[i] == targets_numpy[i]:
        matches += 1
print(f"Количество совпадений: {matches} / {len(binary_predictions)}, {matches / len(binary_predictions) * 100:.2f}%")


accuracy = accuracy_score(targets_numpy, binary_predictions)
precision = precision_score(targets_numpy, binary_predictions, zero_division='warn')
recall = recall_score(targets_numpy, binary_predictions, zero_division='warn')
f1 = f1_score(targets_numpy, binary_predictions, zero_division='warn')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')

epochs = range(1, len(losses) + 1)
val_epoch = range(1, len(val_losses) + 1)

# Построение графика
plt.plot(epochs, losses, marker='o', linestyle='-', label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend()
plt.show()

plt.plot(val_epoch, val_losses, marker='o', linestyle='-', label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Val Loss')
plt.title('Val Loss vs Epochs')
plt.legend()
plt.show()

# Number of finished trials: 100
# Best trial:
# Value:  0.7361963190184049
# Params:
#     learning_rate: 0.04445271213047534
#     num_hidden_units_layer1: 112
#     num_hidden_units_layer2: 24
#     dropout_rate: 0.4370075217810576
#     weight_decay: 0.0002725967716773317

# Number of finished trials: 10
# Best trial:
# Value:  0.7684049079754601
# Params:
#     learning_rate: 0.0016998743176789477
#     num_hidden_units_layer1: 104
#     num_hidden_units_layer2: 112
#     dropout_rate: 0.754657632149016
#     weight_decay: 1.4739545003452742e-05

# Number of finished trials: 50
# Best trial:
# Value:  0.8103448275862069
# Params:
#     learning_rate: 0.0745489190252389
#     num_hidden_units_layer1: 42
#     num_hidden_units_layer2: 51
#     dropout_rate: 0.68869860974142
#     weight_decay: 6.557185922917967e-05
