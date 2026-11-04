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


def train_network(network, X, y, epochs=10, learning_rate=1e-3, batch_size=128):
    torch.manual_seed(42)  # same starting weights and batch order every run
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_train, X_test, y_train, y_test = split_data(X, y)
    # pandas -> tensors on the GPU: pixels as float32, labels as int64 (what the loss expects)
    X_train = torch.tensor(X_train.values, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train.values, dtype=torch.int64, device=device)
    X_test = torch.tensor(X_test.values, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test.values, dtype=torch.int64, device=device)

    network = network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    # cross-entropy: softmax over the 10 scores, then −log(probability of the true digit)
    loss_function = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, epochs + 1):
        # one epoch = one pass over the training images, in shuffled mini-batches of 128
        network.train()
        order = torch.randperm(len(X_train), device=device)
        total_loss = 0.0
        for start in range(0, len(X_train), batch_size):
            batch = order[start:start + batch_size]
            scores = network(X_train[batch])            # predict: 10 scores per image
            loss = loss_function(scores, y_train[batch])  # how wrong
            optimizer.zero_grad()                       # clear the previous batch's gradients
            loss.backward()                             # gradient for every weight
            optimizer.step()                            # move each weight against its gradient
            total_loss += loss.item() * len(batch)

        # validation: no gradients needed, the highest score is the predicted digit
        network.eval()
        with torch.no_grad():
            accuracy = (network(X_test).argmax(dim=1) == y_test).float().mean().item()
        history.append({'epoch': epoch, 'train_loss': total_loss / len(X_train), 'val_accuracy': accuracy})
        print(f"epoch {epoch:2d}  train loss {total_loss / len(X_train):.4f}  validation accuracy {accuracy:.4f}")
    return network, history


if __name__ == "__main__":
    df = get_data_from_csv()
    X, y = get_X_y(df)
    X = divide_by_255(X)

    # model, X_test, y_test = get_model_and_test_data(X, y)
    # accuracy = accuracy_score(y_test, model.predict(X_test))
    # print(f"Accuracy: {accuracy:.4f}")

    network = build_dense_network()
    network, history = train_network(network, X, y, epochs=10, learning_rate=1e-3, batch_size=128)