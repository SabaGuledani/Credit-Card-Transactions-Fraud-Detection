import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler

def load_dataset(split="train"):
    split = lower(split)
    if split=="train":
        df = pd.read_csv("../data/processed/train_ohe.csv")
    elif split =="test":
        df = pd.read_csv("../data/processed/test_ohe.csv")
    return df


def get_num_columns(df):
       dtypes_df = pd.DataFrame(df.dtypes).reset_index()
       num_cols = list(dtypes_df[dtypes_df[0] != bool]["index"].values)
       return num_cols


def get_train_test_splits(df):
       X = df.drop(columns=["is_fraud"])
       y = df["is_fraud"]

       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y)
       return X_train, X_test, y_train, y_test

def scale_data(df,X_train, X_test):
    num_cols = get_num_columns(df)
    num_cols.remove("is_fraud")

    scaler = RobustScaler()

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols]) # scale numerical columns only
    X_test[num_cols] = scaler.fit_transform(X_test[num_cols])
    return scaler
