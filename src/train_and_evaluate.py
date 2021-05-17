# load train and test
# train algorithm
# save metrices and parameters

import argparse
import os
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, \
                            precision_recall_curve, plot_precision_recall_curve, average_precision_score, \
                            accuracy_score, roc_curve, roc_auc_score
import xgboost as xgb
from collections import Counter

from get_data import read_params


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    roc_auc = roc_auc_score(actual, pred)
    return accuracy, roc_auc

def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    n_estimators = config["estimators"]["XGBoost"]["params"]["n_estimators"]
    learning_rate = config["estimators"]["XGBoost"]["params"]["learning_rate"]
    max_depth = config["estimators"]["XGBoost"]["params"]["max_depth"]
    gamma = config["estimators"]["XGBoost"]["params"]["gamma"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    drop_cols = ['ItemDescription', 'OnPromotion_Week1', 'OnPromotion_Week2', 'OnPromotion_Week3', 'OnPromotion_Week4']
    X_train = train.drop(target + drop_cols, axis=1)
    X_test = test.drop(target + drop_cols, axis=1)

    y_train = train[target]
    y_test = test[target]

    n_estimators = eval(n_estimators) 
    learning_rate = eval(learning_rate)
    gamma = eval(gamma)

    # Create an instance of XGBoost
    y_count = y_train.value_counts()
    ratio = (y_count[0] / y_count[1])
    xgb_obj = xgb.XGBClassifier(random_state=random_state, scale_pos_weight=ratio, use_label_encoder=False)
    xgb_param = {'n_estimators': n_estimators,
                 'learning_rate': learning_rate,
                 'max_depth': max_depth,
                 'gamma': gamma,
                }
    # Randomized Serach CV
    xgb_rand = RandomizedSearchCV(xgb_obj, xgb_param, cv=5, scoring='accuracy', refit=True, n_jobs=-1, verbose=5, random_state=random_state)
    # Model fit
    xgb_rand_fit = xgb_rand.fit(X_train, y_train)
    # Best estimator
    xgb_rand_bm = xgb_rand_fit.best_estimator_

    y_test_pred = xgb_rand_bm.predict(X_test)
    (accuracy, roc_auc) = eval_metrics(y_test, y_test_pred)

    print("XGBoost Model - Test")
    print("  Accuracy: %s" % accuracy)
    print("  ROC_AUC: %s" % roc_auc)

    scores_file = config["reports"]["scores"]

    with open(scores_file, "w") as f:
        scores = {
            "accuracy": accuracy,
            "roc_auc": roc_auc
        }
        json.dump(scores, f, indent=4)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(xgb_rand_bm, model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)

