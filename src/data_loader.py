import pandas as pd

def load_train_data(path: str):
    return pd.read_csv(path)

def load_test_data(path: str):
    return pd.read_csv(path)
