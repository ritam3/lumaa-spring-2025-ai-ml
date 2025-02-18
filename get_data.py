import pandas as pd

# reads the the data from the given path
def get_data(path):
    return pd.read_csv(path)