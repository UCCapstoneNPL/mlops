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
from sklearn.ensemble import RandomForestClassifier
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

    n_estimators = config["estimators"]["random_forest"]["params"]["n_estimators"]
    max_features = config["estimators"]["random_forest"]["params"]["max_features"]
    max_depth = config["estimators"]["random_forest"]["params"]["max_depth"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    drop_cols = ['ItemDescription', 'OnPromotion_Week1', 'OnPromotion_Week2', 'OnPromotion_Week3', 'OnPromotion_Week4']
    X_train = train.drop(target + drop_cols, axis=1)
    X_test = test.drop(target + drop_cols, axis=1)

    y_train = train[target]
    y_test = test[target]

    n_estimators = eval(n_estimators) 

    # Create an instance of Random Forest
    rf_obj = RandomForestClassifier(class_weight={0:1, 1:8.3})
 
    # Create param_grid for RandomizedSearchCV
    param_rand = {'n_estimators': n_estimators,
             'max_features': max_features,
             'max_depth': max_depth
             }
             
    # Randomized Serach CV
    rf_rand = RandomizedSearchCV(rf_obj, param_rand, cv=5, scoring='accuracy', refit=True, n_jobs=-1, verbose=5, random_state=random_state)

    # Model fit
    rf_rand_fit = rf_rand.fit(X_train, y_train)
    # Best estimator
    rf_rand_bm = rf_rand_fit.best_estimator_

    y_test_pred = rf_rand_bm.predict(X_test)
    (accuracy, roc_auc) = eval_metrics(y_test, y_test_pred)

    print("Random Forest Model - Test")
    print("  Accuracy: %s" % accuracy)
    print("  ROC_AUC: %s" % roc_auc)

    y_train_pred = rf_rand_bm.predict(X_train)
    (accuracy_train, roc_auc_train) = eval_metrics(y_train, y_train_pred)

    print("Random Forest Model - Train")
    print("  Accuracy: %s" % accuracy_train)
    print("  ROC_AUC: %s" % roc_auc_train)

    scores_file = config["reports"]["scores"]

    with open(scores_file, "w") as f:
        scores = {
            "accuracy": accuracy,
            "roc_auc": roc_auc
        }
        json.dump(scores, f, indent=4)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(rf_rand_bm, model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)

