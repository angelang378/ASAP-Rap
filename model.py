import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def get_input_output(first_csv, second_csv=None):
    category_1 = pd.read_csv(first_csv)
    category_1.head()
    category_1_x = category_1.iloc[:, 5:]
    category_1_y = pd.DataFrame(
        np.full(category_1.shape[0],
                0), columns=['rank'])  # 0 for first category, 1 for second

    category_2 = None

    if second_csv is not None:
        category_2 = pd.read_csv(second_csv)
        category_2.head()
        category_2_x = category_2.iloc[:, 5:]
        category_2_y = pd.DataFrame(np.full(category_2.shape[0], 1),
                                    columns=['rank'])

    if category_2 is None:
        return category_1.iloc[:, 3], category_1.iloc[:, 5:]
    else:
        return pd.concat([category_1_x, category_2_x
                          ]), pd.concat([category_1_y, category_2_y])


## train data
class TrainData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


## test data
class TestData(Dataset):
    def __init__(self, x_data):
        self.x_data = x_data

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return len(self.x_data)


class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(12, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def train_model(x_path, y_path):
    combined_x, combined_y = get_input_output(x_path, y_path)
    x_train, x_test, y_train, y_test = train_test_split(combined_x,
                                                        combined_y,
                                                        test_size=0.33,
                                                        random_state=69)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    EPOCHS = 200
    BATCH_SIZE = 30
    LEARNING_RATE = 0.005

    train_data = TrainData(torch.FloatTensor(x_train),
                           torch.FloatTensor(y_train))

    test_data = TestData(torch.FloatTensor(x_test))

    train_loader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = BinaryClassification()
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()

    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch)
            acc = binary_acc(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        if e % 100 == 0:
            print(
                f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}'
            )

    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    torch.save(model.state_dict(), "trained_models/trained_model.pt")


def predict(data, mpath, user_1, user_2):

    with torch.no_grad():

        # Retrieve song data to predict
        song_titles, beatles_data = get_input_output(data)

        scaler = StandardScaler()
        predict_tensor = scaler.fit_transform(
            torch.from_numpy(beatles_data.values))

        # Loading the saved model
        model = BinaryClassification()
        model.load_state_dict(torch.load(mpath))
        model.eval()

        # Generate prediction
        prediction = model(torch.from_numpy(predict_tensor).float())
        # preds = (torch.round(torch.sigmoid(preds)))

        #ranked list of songs for person 0
        ranked = [x for _, x in sorted(zip(prediction[:, 0], song_titles))]
        print(ranked)

        names = [user_1, user_2]
        preds = np.where(prediction < 0, 0, 1)
        # predictions for each song title
        for i in range(len(preds)):
            print("Song: ", song_titles[i], " | Prediction: ", names[preds[i,
                                                                           0]])

        #total songs classified for each person
        print(names[0], " total: ", np.count_nonzero(preds == 0))
        print(names[1], " total: ", np.count_nonzero(preds == 1))

        return prediction


def main():
    # train_model("data/vivian.csv", "data/william.csv")
    # predict('data/beatles.csv', "trained_models/trained_model.pt", "Vivian", "William")
    predict('data/vivian.csv', "trained_models/trained_model.pt", "Vivian",
            "William")
    # predict('data/william.csv', "trained_models/trained_model.pt","Vivian", "William")


if __name__ == "__main__":
    main()
