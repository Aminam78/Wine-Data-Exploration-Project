# src/utils.py
from sklearn.datasets import load_wine
import pandas as pd

def load_and_prepare_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return df, wine.feature_names, wine.target