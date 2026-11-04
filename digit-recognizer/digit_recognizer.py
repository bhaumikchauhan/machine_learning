import pandas as pd
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


def get_model_and_test_data(X, y):
    model = LogisticRegression(max_iter=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


if __name__ == "__main__":
    df = get_data_from_csv()
    X, y = get_X_y(df)
    X = divide_by_255(X)

    model, X_test, y_test = get_model_and_test_data(X, y)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy: {accuracy:.4f}")