import pandas as pd

def preprocess(df: pd.DataFrame):
    df = df.copy()
    if 'median_house_value' in df.columns:
        y = df['median_house_value']
        X = df.drop(columns=['median_house_value'])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    X = X.select_dtypes(include=['number']).fillna(0)
    return X, y
