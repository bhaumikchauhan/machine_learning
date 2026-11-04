import copy

import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

def get_data_from_csv():
    return pd.read_csv("kaggle\\input\\digit-recognizer\\train.csv")


def get_X_y(df):
    X = df.drop(columns=['label'])
    y = df['label']
    return X, y

def divide_by_255(X):
    return X / 255.0


def split_data(X, y):
    # the one held-out split every model is judged on
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def get_model_and_test_data(X, y):
    model = LogisticRegression(max_iter=1000)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model.fit(X_train, y_train)
    return model, X_test, y_test


# Written by Claude (on request): the first PyTorch training loop.
def build_dense_network():
    return nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))


def to_tensors(X, y, device):
    # pandas -> tensors on the GPU: pixels as float32, labels as int64 (what the loss expects)
    return (torch.tensor(X.values, dtype=torch.float32, device=device),
            torch.tensor(y.values, dtype=torch.int64, device=device))


def accuracy_of(network, X, y):
    # no gradients needed, the highest score is the predicted digit
    network.eval()
    with torch.no_grad():
        return (network(X).argmax(dim=1) == y).float().mean().item()


def train_network(network, X, y, epochs=30, learning_rate=1e-3, batch_size=128):
    torch.manual_seed(42)  # same starting weights and batch order every run
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_train, X_test, y_train, y_test = split_data(X, y)
    # the epoch is chosen on an early-stopping set taken from the training part; the held-out set only judges
    X_fit, X_stop, y_fit, y_stop = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    X_fit, y_fit = to_tensors(X_fit, y_fit, device)
    X_stop, y_stop = to_tensors(X_stop, y_stop, device)
    X_test, y_test = to_tensors(X_test, y_test, device)

    network = network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    # cross-entropy: softmax over the 10 scores, then −log(probability of the true digit)
    loss_function = nn.CrossEntropyLoss()

    history = []
    best_accuracy, best_epoch, best_weights = -1.0, 0, None
    for epoch in range(1, epochs + 1):
        # one epoch = one pass over the training images, in shuffled mini-batches of 128
        network.train()
        order = torch.randperm(len(X_fit), device=device)
        total_loss = 0.0
        for start in range(0, len(X_fit), batch_size):
            batch = order[start:start + batch_size]
            scores = network(X_fit[batch])              # predict: 10 scores per image
            loss = loss_function(scores, y_fit[batch])  # how wrong
            optimizer.zero_grad()                       # clear the previous batch's gradients
            loss.backward()                             # gradient for every weight
            optimizer.step()                            # move each weight against its gradient
            total_loss += loss.item() * len(batch)

        fit_accuracy = accuracy_of(network, X_fit, y_fit)
        stop_accuracy = accuracy_of(network, X_stop, y_stop)
        if stop_accuracy > best_accuracy:
            best_accuracy, best_epoch = stop_accuracy, epoch
            best_weights = copy.deepcopy(network.state_dict())  # a copy, or later epochs would overwrite it
        history.append({'epoch': epoch, 'train_loss': total_loss / len(X_fit),
                        'fit_accuracy': fit_accuracy, 'stop_accuracy': stop_accuracy})
        print(f"epoch {epoch:2d}  train loss {total_loss / len(X_fit):.4f}  "
              f"fit accuracy {fit_accuracy:.4f}  early-stopping accuracy {stop_accuracy:.4f}")

    # keep the best epoch's weights, then look at the held-out images once
    network.load_state_dict(best_weights)
    held_out_accuracy = accuracy_of(network, X_test, y_test)
    print(f"best epoch {best_epoch} (early-stopping accuracy {best_accuracy:.4f}), "
          f"held-out accuracy {held_out_accuracy:.4f}")
    return network, history, held_out_accuracy


if __name__ == "__main__":
    df = get_data_from_csv()
    X, y = get_X_y(df)
    X = divide_by_255(X)

    # model, X_test, y_test = get_model_and_test_data(X, y)
    # accuracy = accuracy_score(y_test, model.predict(X_test))
    # print(f"Accuracy: {accuracy:.4f}")

    network = build_dense_network()
    network, history, held_out_accuracy = train_network(network, X, y, epochs=30, learning_rate=1e-3, batch_size=128)