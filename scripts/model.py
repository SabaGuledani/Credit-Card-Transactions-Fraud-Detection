import pandas as pd
import pickle 
import os
from score_functions import get_probs, print_scores
from get_features import get_features
from preprocessor import prepare_data
from xgboost import XGBClassifier

data_path = "../data/processed/test_features.csv"
if not os.path.exists(data_path):
    raw_data_filepath = "../data/raw/fraudTest.csv"
    df = pd.read_csv(raw_data_filepath)
    df = get_features(df)

    df.to_csv("../data/processed/test_features.csv", index=False)
else:
    df = pd.read_csv("../data/processed/test_features.csv")

preprocessed_path = "../data/processed/test_ohe.csv"
if not os.path.exists(preprocessed_path):
    df = prepare_data(data_path, preprocessed_path)
else:
    df = pd.read_csv(preprocessed_path)

model_path = "xgboost_93.pkl"

if not os.path.exists(model_path):
    print( "Model not found")

model = pickle.load(open(model_path, "rb"))

X_test = df.drop(columns=["is_fraud"])
y_test = df["is_fraud"]

y_pred_prob = get_probs(model, df)
roc_auc, pr_auc, best_recall = print_scores(y_test, y_pred_prob, target_precision=0.95)
