import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import torch
import torch.nn as nn
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from torch import optim, randperm
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, RandomSampler, SubsetRandomSampler
from geochem import *
from func import *
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def draw_results_graphs(loss, epochs):
    fig, axs = plt.subplots(1, 1, figsize=(16, 8))
    epoch = list(range(1, epochs + 1))
    axs.plot(epoch, loss, marker='o', linestyle='-', label='Val Loss')
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Loss')
    axs.set_title('Loss vs Epochs')
    axs.legend()

    fig.suptitle(f'\nTrain Loss Plot: ')
    plt.subplots_adjust(top=0.8)
    plt.show()


class PyTorchClassifier:
    def __init__(self, model, input_dim, output_dim, hidden_units,
                            dropout_rate, activation_function,
                            loss_function, optimizer, learning_rate, weight_decay,
                            epochs, regular, early_stopping, patience, labels, batch_size=20):
        self.model = model(input_dim, output_dim, hidden_units,
                           dropout_rate, activation_function)
        self.criterion = loss_function
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.epochs = epochs
        self.regular = regular
        self.early_stopping = early_stopping
        self.patience = patience
        self.labels = labels
        self.batch_size = batch_size

    def fit(self, X_train, y_train):
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        losses = []
        best_loss = float('inf')
        patience = self.patience

        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0
            indices = torch.arange(X_train.shape[0])
            indices = torch.randperm(len(indices))
            batch_size = 32
            for i in range(0, len(X_train), batch_size):
                start_idx = i
                end_idx = min(start_idx + batch_size, len(X_train))
                inputs = X_train[start_idx:end_idx]
                labels = y_train[start_idx:end_idx].unsqueeze(1)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                l2_lambda = self.regular
                l2_reg = torch.tensor(0.)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += l2_lambda * l2_reg
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {running_loss / (X_train.shape[0] / self.batch_size)}')

            if self.early_stopping:
                if loss < best_loss:
                    best_loss = loss
                    patience = self.patience
                else:
                    patience -= 1
                    if patience == 0:
                        print(f"     Epoch [{epoch}/{self.epochs}] Early stopping")
                        self.epochs = epoch
                        break
            losses.append(running_loss / (X_train.shape[0] / self.batch_size))
        draw_results_graphs(losses, self.epochs)

    def predict(self, X):
        predictions = []
        mark_pred = []
        X = torch.from_numpy(X).float()

        self.model.eval()
        with torch.no_grad():
            pred_batch = self.model(X)
            predictions.extend([np.hstack((pred.numpy(), 1 - pred.numpy())) for pred in pred_batch])
            mark_pred.extend([pred.numpy() for pred in pred_batch])
        mark = [item for m in mark_pred for item in m]
        mark = np.where(np.array(mark) > 0.5, 1, 0)
        labels = self.labels
        label_mark = []
        label_mark.extend([labels[m] for m in mark if m in labels])
        return label_mark

    def predict_proba(self, X):
        predictions = []
        X = torch.from_numpy(X).float()
        with torch.no_grad():
            pred_batch = self.model(X)
            predictions.extend([np.hstack((pred.numpy(), 1 - pred.numpy())) for pred in pred_batch])
        return predictions


    def metrics(self, X_val, y_val, opt_threshold=0.5):
        all_targets = []
        all_predictions = []
        predictions = []
        X = torch.from_numpy(X_val).float()
        self.model.eval()
        with torch.no_grad():
            val_dl = DataLoader(TensorDataset(X, y_val), batch_size=20, shuffle=False)
            for xb, yb in val_dl:
                pred_batch = self.model(xb)
                all_targets.extend([y.numpy() for y in yb])
                all_predictions.extend([pred.numpy() for pred in pred_batch])
                predictions.extend([np.hstack((pred.numpy(), 1 - pred.numpy())) for pred in pred_batch])
        all_predictions = [1 if p >= opt_threshold else 0 for p in all_predictions]
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        return accuracy, precision, recall, f1

def draw_roc_curve(y_val, y_mark):
    fpr, tpr, thresholds = roc_curve(y_val, y_mark)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def classify_based_on_roc(y_val, y_mark, threshold_strategy="accuracy"):
    fpr, tpr, thresholds = roc_curve(y_val, y_mark)
    if threshold_strategy == "accuracy":
        accuracy = tpr + (1 - fpr)
        opt_idx = np.argmax(accuracy)
    elif threshold_strategy == "sensitivity":
        opt_idx = np.argmax(tpr)
    elif threshold_strategy == "specificity":
        tnr = 1 - fpr
        opt_idx = np.argmax(tnr)

    opt_threshold = thresholds[opt_idx]
    mark = np.where(np.array(y_mark) > opt_threshold, 1, 0)
    return opt_threshold, mark

def torch_classifier_train():
    TorchClassifier = QtWidgets.QDialog()
    ui_tch = Ui_TorchClassifierForm()
    ui_tch.setupUi(TorchClassifier)
    TorchClassifier.show()
    TorchClassifier.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    data, list_param = build_table_train(True, 'mlp')
    list_param_mlp = data.columns.tolist()[2:]
    labels = set_marks()
    labels_dict = {value: key for key, value in labels.items()}
    data['mark'] = data['mark'].replace(labels)
    data = data.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 2:], data['mark'], test_size=0.2, random_state=42)
    y_train = y_train.values
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    X_val = X_test.values
    y_val = torch.from_numpy(y_test.values).float()

    def torch_classifier_lineup():
        input_dim = X_train.shape[1]
        output_dim = 1

        epochs = ui_tch.spinBox_epochs.value()
        learning_rate = ui_tch.doubleSpinBox_choose_lr.value()
        hidden_units = list(map(int, ui_tch.lineEdit_choose_layers.text().split()))
        dropout_rate = ui_tch.doubleSpinBox_choose_dropout.value()
        weight_decay = ui_tch.doubleSpinBox_choose_decay.value()
        regular = ui_tch.doubleSpinBox_choose_reagular.value()

        if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
            activation_function = nn.ReLU()
        elif ui_tch.comboBox_activation_func.currentText() == 'Sigmoid':
            activation_function = nn.Sigmoid()
        elif ui_tch.comboBox_activation_func.currentText() == 'Tanh':
            activation_function = nn.Tanh()

        if ui_tch.comboBox_optimizer.currentText() == 'Adam':
            optimizer = torch.optim.Adam
        elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
            optimizer = torch.optim.SGD

        if ui_tch.comboBox_loss.currentText() == 'CrossEntropy':
            loss_function = nn.CrossEntropyLoss()
        elif ui_tch.comboBox_loss.currentText() == 'BCEWithLogitsLoss':
            loss_function = nn.BCEWithLogitsLoss()
        elif ui_tch.comboBox_loss.currentText() == 'BCELoss':
            loss_function = nn.BCELoss()

        early_stopping = False
        threshold = False
        patience = 0
        if ui_tch.checkBox_early_stop.isChecked():
            early_stopping = True
            patience = ui_tch.spinBox_stop_patience.value()
        if ui_tch.checkBox_threshold.isChecked():
            threshold = True

        pipeline = Pipeline([
            ('features', FeatureUnion([
                ('scaler', StandardScaler())
            ])),
            ('classifier', PyTorchClassifier(Model, input_dim, output_dim, hidden_units,
                                             dropout_rate, activation_function,
                                             loss_function, optimizer, learning_rate, weight_decay,
                                             epochs, regular, early_stopping, patience, labels_dict, batch_size=20))
        ])

        model_name = 'torch_NN_cls'
        text_model = model_name + ' StandardScaler'

        except_mlp = session.query(ExceptionMLP).filter_by(analysis_id=get_MLP_id()).first()
        new_lineup = LineupTrain(
            type_ml='cls',
            analysis_id=get_MLP_id(),
            list_param=json.dumps(list_param_mlp),
            list_param_short=json.dumps(list_param),
            except_signal=except_mlp.except_signal,
            except_crl=except_mlp.except_crl,
            text_model=text_model,
            model_name=model_name,
            over_sampling='none',
            pipe=pickle.dumps(pipeline),
            random_seed=ui.spinBox_seed.value(),
            cvw=False
        )
        session.add(new_lineup)
        session.commit()

        set_info(f'Модель {model_name} добавлена в очередь\n{text_model}', 'green')

    def train():
        input_dim = X_train.shape[1]
        output_dim = 1

        epochs = ui_tch.spinBox_epochs.value()
        learning_rate = ui_tch.doubleSpinBox_choose_lr.value()
        hidden_units = list(map(int, ui_tch.lineEdit_choose_layers.text().split()))
        dropout_rate = ui_tch.doubleSpinBox_choose_dropout.value()
        weight_decay = ui_tch.doubleSpinBox_choose_decay.value()
        regular = ui_tch.doubleSpinBox_choose_reagular.value()

        if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
            activation_function = nn.ReLU()
        elif ui_tch.comboBox_activation_func.currentText() == 'Sigmoid':
            activation_function = nn.Sigmoid()
        elif ui_tch.comboBox_activation_func.currentText() == 'Tanh':
            activation_function = nn.Tanh()

        if ui_tch.comboBox_optimizer.currentText() == 'Adam':
            optimizer = torch.optim.Adam
        elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
            optimizer = torch.optim.SGD

        if ui_tch.comboBox_loss.currentText() == 'CrossEntropy':
            loss_function = nn.CrossEntropyLoss()
        elif ui_tch.comboBox_loss.currentText() == 'BCEWithLogitsLoss':
            loss_function = nn.BCEWithLogitsLoss()
        elif ui_tch.comboBox_loss.currentText() == 'BCELoss':
            loss_function = nn.BCELoss()

        early_stopping = False
        threshold = False
        patience = 0
        if ui_tch.checkBox_early_stop.isChecked():
            early_stopping = True
            patience = ui_tch.spinBox_stop_patience.value()
        if ui_tch.checkBox_threshold.isChecked():
            threshold = True

        pipeline = Pipeline([
            ('features', FeatureUnion([
                ('scaler', StandardScaler())
            ])),
            ('classifier', PyTorchClassifier(Model, input_dim, output_dim, hidden_units,
                                             dropout_rate, activation_function,
                                             loss_function, optimizer, learning_rate, weight_decay,
                                             epochs, regular, early_stopping, patience, labels_dict, batch_size=20))
        ])

        start_time = datetime.datetime.now()
        pipeline.fit(X, y)

        y_mark = pipeline.predict(X_val)
        mark = []
        mark.extend([labels[m] for m in y_mark if m in labels])
        y_res = pipeline.predict_proba(X_val)
        y_prob = [i[0] for i in y_res]

        accuracy = accuracy_score(y_val, mark)
        print('Accuracy: ', accuracy)
        draw_roc_curve(y_val, y_prob)
        end_time = datetime.datetime.now() - start_time
        print(end_time)

        if ui_tch.checkBox_save_model.isChecked():
            text_model = '*** TORCH NN *** \n' + 'test_accuray: ' + str(round(accuracy, 3)) + '\nвремя обучения: ' \
                         + str(end_time) + '\nlearning_rate: ' + str(learning_rate) + '\nhidden units: ' + str(
                hidden_units) \
                         + '\nweight decay: ' + str(weight_decay) + '\ndropout rate: ' + str(dropout_rate) + \
                         '\nregularization: ' + str(regular)
            torch_save_classifier(pipeline, accuracy, list_param, text_model)
            print('Model saved')

    def tune_params():
        start_time = datetime.datetime.now()
        input_dim = X_train.shape[1]
        output_dim = 1

        epochs = ui_tch.spinBox_epochs.value()
        learning_rate = str_to_interval(ui_tch.lineEdit_tune_lr.text())
        dropout_rate = str_to_interval(ui_tch.lineEdit_tune_dropout.text())
        weight_decay = str_to_interval(ui_tch.lineEdit_tune_decay.text())
        regularization = ui_tch.doubleSpinBox_choose_reagular.value()
        hidden_units = str_to_interval(ui_tch.lineEdit_tune_layers.text())
        layers_num = ui_tch.spinBox_layers_num.value()
        patience = ui_tch.spinBox_stop_patience.value()

        if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
            activation_function = nn.ReLU()
        elif ui_tch.comboBox_activation_func.currentText() == 'Sigmoid':
            activation_function = nn.Sigmoid()
        elif ui_tch.comboBox_activation_func.currentText() == 'Tanh':
            activation_function = nn.Tanh()

        if ui_tch.comboBox_optimizer.currentText() == 'Adam':
            optimizer = torch.optim.Adam
        elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
            optimizer = torch.optim.SGD

        if ui_tch.comboBox_loss.currentText() == 'CrossEntropy':
            loss_function = nn.CrossEntropyLoss()
        elif ui_tch.comboBox_loss.currentText() == 'BCEWithLogitsLoss':
            loss_function = nn.BCEWithLogitsLoss()
        elif ui_tch.comboBox_loss.currentText() == 'BCELoss':
            loss_function = nn.BCELoss()

        early_stopping = False
        if ui_tch.checkBox_early_stop.isChecked():
            early_stopping = True


        def objective(trial):
            print("Trial number:", trial.number)
            op_learning_rate = trial.suggest_float('learning_rate', learning_rate[0], learning_rate[1], log=True)
            op_num_hidden_units = [trial.suggest_int(f'num_hidden_units_layer{i}',
                                                     hidden_units[0], hidden_units[1]) for i in range(1, layers_num)]
            op_dropout_rate = trial.suggest_float('dropout_rate', dropout_rate[0], dropout_rate[1], log=True)
            op_weight_decay = trial.suggest_float('weight_decay', weight_decay[0], weight_decay[1], log=True)
            op_regularization = trial.suggest_float('regularization', regularization, regularization + 0.1, log=True)

            pipeline = Pipeline([
                ('features', FeatureUnion([
                    ('scaler', StandardScaler())
                ])),
                ('classifier', PyTorchClassifier(Model, input_dim, output_dim, op_num_hidden_units,
                                                 op_dropout_rate, activation_function,
                                                 loss_function, optimizer, op_learning_rate, op_weight_decay,
                                                 epochs, op_regularization, early_stopping, patience,
                                                    labels_dict, batch_size=20))
            ])

            pipeline.fit(X, y)

            y_mark = pipeline.predict(X_val)
            mark = []
            mark.extend([labels[m] for m in y_mark if m in labels])
            accuracy = accuracy_score(y_val, mark)
            print('Accuracy: ', accuracy)
            return accuracy

        num_trials = ui_tch.spinBox_trials.value()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize',
                                    pruner=optuna.pruners.MedianPruner(
                                        n_startup_trials=2, n_warmup_steps=20, interval_steps=1
                                    ), sampler=optuna.samplers.RandomSampler(seed=10))
        study.optimize(objective, n_trials=num_trials)

        print("Number of finished trials:", len(study.trials))
        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)
        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        best_learning_rate = trial.params['learning_rate']
        best_num_hidden_units = [trial.params[f'num_hidden_units_layer{i}'] for i in range(1, layers_num)]
        best_dropout_rate = trial.params['dropout_rate']
        best_weight_decay = trial.params['weight_decay']
        best_regularization = trial.params['regularization']

        epochs = ui_tch.spinBox_epochs.value()
        if ui_tch.comboBox_activation_func.currentText() == 'ReLU':
            activation_function = nn.ReLU()
        elif ui_tch.comboBox_activation_func.currentText() == 'Sigmoid':
            activation_function = nn.Sigmoid()
        elif ui_tch.comboBox_activation_func.currentText() == 'Tanh':
            activation_function = nn.Tanh()

        if ui_tch.comboBox_optimizer.currentText() == 'Adam':
            optimizer = torch.optim.Adam
        elif ui_tch.comboBox_optimizer.currentText() == 'SGD':
            optimizer = torch.optim.SGD

        if ui_tch.comboBox_loss.currentText() == 'BCEWithLogitsLoss':
            loss_function = torch.nn.BCEWithLogitsLoss()
        elif ui_tch.comboBox_loss.currentText() == 'CrossEntropy':
            loss_function = torch.nn.CrossEntropyLoss()
        elif ui_tch.comboBox_loss.currentText() == 'BCELoss':
            loss_function = torch.nn.BCELoss()

        early_stopping = False
        if ui_tch.checkBox_early_stop.isChecked():
            early_stopping = True

        pipeline = Pipeline([
            ('features', FeatureUnion([
                ('scaler', StandardScaler())
            ])),
            ('classifier', PyTorchClassifier(Model, input_dim, output_dim, best_num_hidden_units,
                                             best_dropout_rate, activation_function,
                                             loss_function, optimizer, best_learning_rate, best_weight_decay,
                                             epochs, best_regularization, early_stopping, patience,
                                             labels_dict, batch_size=20))
        ])

        pipeline.fit(X, y)
        y_mark = pipeline.predict(X_val)
        mark = []
        mark.extend([labels[m] for m in y_mark if m in labels])
        accuracy = accuracy_score(y_val, mark)
        print('Best Accuracy: ', accuracy)
        end_time = datetime.datetime.now() - start_time
        print(end_time)

        if ui_tch.checkBox_save_model.isChecked():
            text_model = '*** TORCH NN *** \n' + 'test_accuray: ' + str(round(accuracy, 3)) + '\nвремя обучения: ' \
                         + str(end_time) + '\nlearning_rate: ' + str(best_learning_rate) + '\nhidden units: ' + str(
                best_num_hidden_units) \
                         + '\nweight decay: ' + str(best_weight_decay) + '\ndropout rate: ' + str(best_dropout_rate) + \
                         '\nregularization: ' + str(best_regularization)
            torch_save_classifier(pipeline, accuracy, list_param, text_model)
            print('Model saved')

    def choose():
        if ui_tch.checkBox_choose_param.isChecked():
            train()

        if ui_tch.checkBox_tune_param.isChecked():
            tune_params()

    ui_tch.pushButton_lineup.clicked.connect(torch_classifier_lineup)
    ui_tch.pushButton_train.clicked.connect(choose)
    TorchClassifier.exec()
