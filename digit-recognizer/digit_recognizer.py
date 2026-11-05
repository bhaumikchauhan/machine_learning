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

def build_conv_network():
    unflatten_layer = nn.Unflatten(1, (1, 28, 28))  # the 784 pixels are a single-channel 28x28 image
    return nn.Sequential(
        unflatten_layer,
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        nn.Linear(128, 10)
    )


def to_tensors(X, y, device):
    # pandas -> tensors on the GPU: pixels as float32, labels as int64 (what the loss expects)
    return (torch.tensor(X.values, dtype=torch.float32, device=device),
            torch.tensor(y.values, dtype=torch.int64, device=device))


def accuracy_of(network, X, y):
    # no gradients needed, the highest score is the predicted digit
    network.eval()
    with torch.no_grad():
        return (network(X).argmax(dim=1) == y).float().mean().item()


def train_network(build_network, X, y, epochs=30, learning_rate=1e-3, batch_size=128, seed=42,
                  early_stopping=True, verbose=True, held_out=True):
    # the network is built here, after the seed, so the seed fixes its starting weights too
    torch.manual_seed(seed)
    # cuDNN's fastest convolution algorithms add in a varying order, so a CNN would differ run to run without this
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = build_network().to(device)
    if held_out:
        X_train, X_test, y_train, y_test = split_data(X, y)
    else:
        # for the submission: no held-out set, every labelled image is used for training (and early stopping)
        X_train, y_train = X, y
        X_test, y_test = None, None
    if early_stopping:
        # the epoch is chosen on an early-stopping set taken from the training part; the held-out set only judges
        X_fit, X_stop, y_fit, y_stop = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
        X_stop, y_stop = to_tensors(X_stop, y_stop, device)
    else:
        # train on the whole training part and keep the last epoch
        X_fit, y_fit = X_train, y_train
        X_stop, y_stop = None, None
    X_fit, y_fit = to_tensors(X_fit, y_fit, device)

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
        stop_accuracy = accuracy_of(network, X_stop, y_stop) if early_stopping else None
        if early_stopping and stop_accuracy > best_accuracy:
            best_accuracy, best_epoch = stop_accuracy, epoch
            best_weights = copy.deepcopy(network.state_dict())  # a copy, or later epochs would overwrite it
        history.append({'epoch': epoch, 'train_loss': total_loss / len(X_fit),
                        'fit_accuracy': fit_accuracy, 'stop_accuracy': stop_accuracy})
        if verbose:
            line = f"epoch {epoch:2d}  train loss {total_loss / len(X_fit):.4f}  fit accuracy {fit_accuracy:.4f}"
            if early_stopping:
                line += f"  early-stopping accuracy {stop_accuracy:.4f}"
            print(line)

    # keep the best epoch's weights (or the last epoch's), then look at the held-out images once
    if early_stopping:
        network.load_state_dict(best_weights)
    held_out_accuracy = accuracy_of(network, *to_tensors(X_test, y_test, device)) if held_out else None
    if verbose:
        chosen = f"best epoch {best_epoch} (early-stopping accuracy {best_accuracy:.4f})" if early_stopping \
            else f"last epoch {epochs}"
        print(f"{chosen}, held-out accuracy {held_out_accuracy:.4f}" if held_out else chosen)
    return network, history, held_out_accuracy


# Written by Claude: the Kaggle submission.
def get_kaggle_test_data_from_csv():
    return pd.read_csv("kaggle\\input\\digit-recognizer\\test.csv")


def predict_digits(network, X):
    # the highest of the 10 scores is the predicted digit
    device = next(network.parameters()).device
    network.eval()
    with torch.no_grad():
        X = torch.tensor(X.values, dtype=torch.float32, device=device)
        return network(X).argmax(dim=1).cpu().numpy()


def create_submission(path='submission.csv'):
    # The CNN trained on all 42,000 labelled images: 33,600 to train on, 8,400 to choose the epoch (best of 30)
    X, y = get_X_y(get_data_from_csv())
    network, _, _ = train_network(build_conv_network, divide_by_255(X), y, held_out=False)
    X_test = divide_by_255(get_kaggle_test_data_from_csv())
    labels = predict_digits(network, X_test)
    pd.DataFrame({'ImageId': range(1, len(labels) + 1), 'Label': labels}).to_csv(path, index=False)
    print(f"Saved {path}: {len(labels)} rows, share per digit "
          f"{pd.Series(labels).value_counts(normalize=True).sort_index().round(3).to_dict()}")


if __name__ == "__main__":
    df = get_data_from_csv()
    X, y = get_X_y(df)
    X = divide_by_255(X)

    # model, X_test, y_test = get_model_and_test_data(X, y)
    # accuracy = accuracy_score(y_test, model.predict(X_test))
    # print(f"Accuracy: {accuracy:.4f}")

    network, history, held_out_accuracy = train_network(build_conv_network, X, y, epochs=30, learning_rate=1e-3,
                                                        batch_size=128)