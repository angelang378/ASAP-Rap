import numpy as np
from numpy.core.records import record
import pandas as pd
import re
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_input_output(first_csv, second_csv=None, user_1=None, user_2=None):
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
        # only add new artists when training model; for predictions, just use curr artist list
        record_artists(category_1.iloc[:, 1].drop_duplicates(), user_1, user_2)
        record_artists(category_2.iloc[:, 1].drop_duplicates(), user_1, user_2)

    # add artists to matrix
    artists = pd.read_csv('data/artists_{}_{}.csv'.format(user_1, user_2),
                          sep='_').to_numpy().flatten()
    artist_matrix_1 = pd.DataFrame(np.full((category_1.shape[0], len(artists)),
                                           0),
                                   columns=artists)

    if category_2 is None:
        for a in artists:
            artist_matrix_1[a] = np.where(category_1['artist'] == a, 1, 0)
            matrix1 = category_1_x.join(artist_matrix_1)
        return category_1.iloc[:, 3], matrix1
    else:
        artist_matrix_2 = pd.DataFrame(np.full(
            (category_2.shape[0], len(artists)), 0),
                                       columns=artists)
        for a in artists:
            artist_matrix_1[a] = np.where(category_1['artist'] == a, 1, 0)
            artist_matrix_2[a] = np.where(category_2['artist'] == a, 1, 0)

        matrix1 = category_1_x.join(artist_matrix_1)
        matrix2 = category_2_x.join(artist_matrix_2)
        return pd.concat([matrix1,
                          matrix2]), pd.concat([category_1_y, category_2_y])


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


# from Munkhdalai et al. 2020
class BinaryClassification(nn.Module):
    def __init__(self, num_input_features):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(num_input_features, 32)
        self.layer_2 = nn.Linear(32, 16)
        self.layer_3 = nn.Linear(16, 8)
        self.layer_4 = nn.Linear(8, 32)
        self.layer_out = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.batchnorm2 = nn.BatchNorm1d(16)
        self.batchnorm3 = nn.BatchNorm1d(8)
        self.batchnorm4 = nn.BatchNorm1d(32)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.relu(self.layer_4(x))
        x = self.batchnorm4(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def train_model(x_path,
                y_path,
                user_1,
                user_2,
                epochs=350,
                learning_rate=0.003,
                freq=50):
    combined_x, combined_y = get_input_output(x_path, y_path, user_1, user_2)
    x_train, x_test, y_train, y_test = train_test_split(combined_x,
                                                        combined_y,
                                                        test_size=0.33,
                                                        random_state=69)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    EPOCHS = epochs
    BATCH_SIZE = 30
    LEARNING_RATE = learning_rate
    FREQ = freq

    err_tr = []
    # use below when testing epochs
    # print_list = [epochs]
    # use below when testing lr, or in general
    print_list = np.arange(1, EPOCHS + 1, FREQ)
    if (EPOCHS not in print_list):
        print_list = list(print_list)
        print_list.append(EPOCHS)  #print last epoch results

    train_data = TrainData(torch.FloatTensor(x_train),
                           torch.FloatTensor(y_train))

    test_data = TestData(torch.FloatTensor(x_test))

    train_loader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = BinaryClassification(x_train.shape[1])
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
        if e in print_list:
            err_tr.append(epoch_loss / len(train_loader))
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

    # y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    test_results = (np.asarray(y_pred_list).flatten() != y_test.flatten())
    err_te = test_results.sum() / test_results.size

    # save_data(model, user_1, user_2)

    return err_tr, err_te


def record_artists(artists, user_1, user_2):
    curr = []
    try:
        data = pd.read_csv("data/artists_{}_{}.csv".format(user_1, user_2),
                           sep='_')
        curr = data.iloc[:, 0].to_numpy()
    except IOError:
        file = open('data/artists_{}_{}.csv'.format(user_1, user_2), 'a')
        file.write("artist")
        file.close()

    new_artists = np.setdiff1d(artists, curr)

    file = open('data/artists_{}_{}.csv'.format(user_1, user_2), 'a')

    for a in new_artists:
        file.write("\n{}".format(a))
    file.close()


def save_data(model, user_1, user_2):
    file_data = get_models()
    lines = len(file_data)
    model_name = "trained_model{}".format(lines)
    torch.save(model.state_dict(),
               "trained_models/trained_model{}.pt".format(lines))
    file_map = open('trained_models/map.csv', 'a')
    file_map.write("\n{},{},{}".format(model_name, user_1, user_2))
    file_map.close()


def get_users(mpath):
    m = get_models()
    num = int(re.search("\d", mpath).group())
    data = m.iloc[num, 1:]
    return data[0], data[1]


def get_models():
    file_map = pd.read_csv("trained_models/map.csv")
    return file_map


def predict(data, mpath):

    with torch.no_grad():

        user_1, user_2 = get_users(mpath)
        # Retrieve song data to predict
        song_titles, song_data = get_input_output(data,
                                                  user_1=user_1,
                                                  user_2=user_2)

        scaler = StandardScaler()
        predict_tensor = scaler.fit_transform(
            torch.from_numpy(song_data.values))

        # Loading the saved model
        model = BinaryClassification(predict_tensor.shape[1])
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
        songs = []
        for i in range(len(preds)):
            print("Song: ", song_titles[i], " | Prediction: ", names[preds[i,
                                                                           0]])
            songs.append([song_titles[i], names[preds[i, 0]]])

        song_recs = pd.DataFrame(songs, columns=['Song', 'User'])

        #total songs classified for each person
        print(names[0], " total: ", np.count_nonzero(preds == 0))
        print(names[1], " total: ", np.count_nonzero(preds == 1))

        return song_recs


def generate_lr_graphs(err_tr, print_list):
    plt.figure()
    line_1, = plt.plot(print_list, err_tr[0], label="LR = 0.001")
    line_2, = plt.plot(print_list, err_tr[1], label="LR = 0.003")
    line_3, = plt.plot(print_list, err_tr[2], label="LR = 0.005")
    line_4, = plt.plot(print_list, err_tr[3], label="LR = 0.007")
    line_5, = plt.plot(print_list, err_tr[4], label="LR = 0.009")
    plt.title("Model 4 Training Loss Over Epochs")
    plt.legend(handles=[line_1, line_2, line_3, line_4, line_5])
    plt.xlabel("# of epochs")
    plt.ylabel("loss")
    plt.show()


def test_learning_rate(epochs, freq):
    learning_rates = [0.001, 0.003, 0.005, 0.007, 0.009]
    errs_tr = []
    errs_te = []
    print_list = np.arange(1, epochs + 1, freq)
    if (epochs not in print_list):
        print_list = list(print_list)
        print_list.append(epochs)

    for i in learning_rates:
        err_tr, err_te = train_model("data/4a6u0ZVG0FWYAJHVggnHAh.csv",
                                     "data/1ukuCLLRLSXE7WYWlbEq2n.csv",
                                     "Vivian", "William", epochs, i, freq)
        errs_tr.append(err_tr)
        errs_te.append(err_te)

    print(errs_te)
    generate_lr_graphs(errs_tr, print_list)


def generate_epoch_graphs(err_tr, err_te, print_list):
    plt.figure()
    line_1, = plt.plot(print_list, err_tr, label="Training error")
    line_2, = plt.plot(print_list, err_te, label="Testing error")
    plt.title("Model 4 Testing vs Training Loss Over Epochs")
    plt.legend(handles=[line_1, line_2])
    plt.xlabel("# of epochs")
    plt.ylabel("loss")
    plt.show()


def test_epochs(learning_rate=0.003):
    epochs = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    errs_tr = []
    errs_te = []
    for i in epochs:
        err_tr, err_te = train_model("data/4a6u0ZVG0FWYAJHVggnHAh.csv",
                                     "data/1ukuCLLRLSXE7WYWlbEq2n.csv",
                                     "Vivian", "William", i, learning_rate, i)
        errs_tr.append(err_tr)
        errs_te.append(err_te)

    print(errs_te)
    generate_epoch_graphs(errs_tr, errs_te, epochs)


def main():
    train_model("data/4a6u0ZVG0FWYAJHVggnHAh.csv",
                "data/1ukuCLLRLSXE7WYWlbEq2n.csv", "Vivian", "William")
    predict('data/1Gf0v4DneJjq3adPSiNVe6.csv',
            "trained_models/trained_model18.pt")


if __name__ == "__main__":
    main()
