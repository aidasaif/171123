
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


def nn_torch(ui_tch, data, list_param, labels, labels_mark):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 2:], data['mark'], test_size=0.2, random_state=42)
    y_train = y_train.values
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    X_val = X_test.values
    y_val = torch.from_numpy(y_test.values).float()

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
                            epochs, regular, early_stopping, patience, labels, batch_size=20))
    ])

    start_time = datetime.datetime.now()
    pipeline.fit(X, y)
    y_mark = pipeline.predict(X_val)
    mark = []
    mark.extend([labels_mark[m] for m in y_mark if m in labels_mark])
    # threshold_strategy = ui_tch.comboBox_threshold.currentText()
    # if threshold:
    #     opt_threshold, mark = classify_based_on_roc(y_val, y_mark, threshold_strategy=threshold_strategy)
    # else:
    #     mark = np.where(np.array(y_mark) > 0.5, 1, 0)
    #     opt_threshold = 0.5

    accuracy = accuracy_score(y_val, mark)
    print('Accuracy: ', accuracy)
    draw_roc_curve(y_val, mark)
    end_time = datetime.datetime.now() - start_time
    print(end_time)

    if ui_tch.checkBox_save_model.isChecked():
        text_model = '*** TORCH NN *** \n' + 'test_accuray: ' + str(round(accuracy, 3)) + '\nвремя обучения: ' \
                     + str(end_time) + '\nlearning_rate: ' + str(learning_rate) + '\nhidden units: ' + str(hidden_units) \
                     + '\nweight decay: ' + str(weight_decay) + '\ndropout rate: ' + str(dropout_rate) + \
                     '\nregularization: ' + str(regular)
        torch_save_classifier(pipeline, accuracy, list_param, text_model)
        print('Model saved')

def set_marks():
    list_cat = [i.title for i in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all()]
    labels = {}
    labels[list_cat[0]] = 1
    labels[list_cat[1]] = 0
    if len(list_cat) > 2:
        for index, i in enumerate(list_cat[2:]):
            labels[i] = index
    return labels


def torch_classifier_train():
    TorchClassifier = QtWidgets.QDialog()
    ui_tch = Ui_TorchClassifierForm()
    ui_tch.setupUi(TorchClassifier)
    TorchClassifier.show()
    TorchClassifier.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    data, list_param = build_table_train(True, 'mlp')
    labels = set_marks()
    labels_dict = {value: key for key, value in labels.items()}
    data['mark'] = data['mark'].replace(labels)
    data = data.fillna(0)

    def train():
        nn_torch(ui_tch, data, list_param, labels_dict, labels)
    def cv():
        nn_cross_val(ui_tch, data)

    ui_tch.pushButton_train.clicked.connect(train)
    ui_tch.pushButton_cross_val.clicked.connect(cv)
    TorchClassifier.exec()
