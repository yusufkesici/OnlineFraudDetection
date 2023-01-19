import pandas as pd
import numpy as np
from helpers.eda import *
from helpers.data_prep import *
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, \
    AdaBoostRegressor, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor, KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import missingno as msno

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def OnlineFraud_DataPrep(df):
    df.drop("isFlaggedFraud", axis=1, inplace=True)
    df.drop(["nameDest", "nameOrig"], axis=1, inplace=True)

    # Feature Engineering
    df["oldOrg/Amount"] = df["oldbalanceOrg"] / (df["amount"] + 0.01)
    df["oldOrg*Amount"] = df["oldbalanceOrg"] * df["amount"]

    df.loc[(df["step"] >= 50) & (df["step"] <= 90), "StepInterval5090"] = 1
    df.loc[(df["step"] < 50) | (df["step"] > 90), "StepInterval5090"] = 0

    df.loc[df["step"] >= 720, "stepInterval720"] = 1
    df.loc[~(df["step"] >= 720), "stepInterval720"] = 0

    df.loc[(df["oldbalanceOrg"] > 550000) & (df["step"] > 430), "StepOldOrgInterval"] = 1
    df.loc[~((df["oldbalanceOrg"] > 550000) & (df["step"] > 430)), "StepOldOrgInterval"] = 0

    df["neworig/old"] = df["newbalanceOrig"] / (df["oldbalanceOrg"] + 0.01)
    df["old*neworig"] = df["oldbalanceOrg"] * df["newbalanceOrig"] + 0.01

    df["oldOrgIsZero"] = np.where(df["oldbalanceOrg"] == 0, 1, 0)
    df["newOrgIsZero"] = np.where(df["newbalanceOrig"] == 0, 1, 0)
    df["newDestIsZero"] = np.where(df["newbalanceDest"] == 0, 1, 0)
    df["olsDestIsZero"] = np.where(df["oldbalanceDest"] == 0, 1, 0)

    df["newOrg/amount"] = df["newbalanceOrig"] / (df["amount"] + 0.01)

    df["newOldDestMultiply"] = df["oldbalanceDest"] * df["newbalanceDest"]

    df["oldDest*Amount"] = df["oldbalanceDest"] * df["amount"]

    df["newDest/amount"] = df["newbalanceDest"] / (df["amount"] + 0.1)

    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    cat_cols.remove("isFraud")

    df = one_hot_encoder(df, cat_cols, True)

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    return X, y


df = pd.read_csv("OnlineFraudDetection/onlineFraudDetection.csv")

X, y = OnlineFraud_DataPrep(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from imblearn.under_sampling import RandomUnderSampler

ranUnSample = RandomUnderSampler()
X_ranUnSample, y_ranUnSample = ranUnSample.fit_resample(X_train, y_train)


def base_models(X, y):
    print("Base Models....")
    regressors = [('LR', LogisticRegression()),
                  ('KNN', KNeighborsClassifier()),
                  ("CART", DecisionTreeClassifier()),
                  ("RF", RandomForestClassifier()),
                  ('Ridge', RidgeClassifier()),
                  ('GBM', GradientBoostingClassifier()),
                  ('XGBoost', XGBClassifier(objective='reg:squarederror')),
                  ('LightGBM', LGBMClassifier()),
                  # ('CatBoost', CatBoostClassifier(verbose=False))
                  ]

    scoring = ["accuracy", "precision", "recall", "f1"]
    for name, regressor in regressors:
        print(f"########### {name} ###############")
        cv_results = cross_validate(regressor, X, y, cv=3, scoring=scoring)
        print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
        print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
        print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
        print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")
        print(f"##################################", end="\n")


base_models(X_ranUnSample, y_ranUnSample)

# Selected Model CART, RF, LGBM


rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "sqrt"],
             "min_samples_split": [2, 15, 20],
             "n_estimators": [100,200, 300]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500, 1000, 5000,10000],
                   "subsample": [1.0, 0.9, 0.5, 0.3]}

CART_params = {'max_depth': [2, 3, 5, 10, 20],
               'min_samples_leaf': [5, 10, 20, 50, 100],
               'criterion': ["gini", "entropy"]}

regressors = [("CART", DecisionTreeClassifier(), CART_params),
              ('LGBM', LGBMClassifier(), lightgbm_params),
              ('RF', RandomForestClassifier(), rf_params),
              ]

best_models = {}
scoring = ['precision', 'recall', 'f1', "accuracy"]
for name, regressor, params in regressors:
    print(f"########## {name} ##########")

    cv_results = cross_validate(regressor, X_ranUnSample, y_ranUnSample, cv=3, scoring=scoring)
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X_ranUnSample, y_ranUnSample)
    final_model = regressor.set_params(**gs_best.best_params_)
    print(f"########## After GridSearchCV Final Results ##########", end="\n")

    cv_results = cross_validate(final_model, X_ranUnSample, y_ranUnSample, cv=3, scoring=scoring)
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")
    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
    best_models[name] = final_model



######################################################
# 5. Stacking & Ensemble Learning
######################################################
scoring = ['precision', 'recall', 'f1', "accuracy"]

def voting_classifier(best_models, X, y):
    print("Voting Regressor...")

    voting_clf = VotingClassifier(estimators=[('LGBM', best_models["LGBM"]),
                                              ('RF', best_models["RF"]),
                                              ('CART', best_models["CART"])]).fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=scoring)
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")
    return voting_clf


voting_clf = voting_classifier(best_models, X_ranUnSample, y_ranUnSample)

######################################################
# 6. Prediction for a New Observation
######################################################
import joblib

y_pred = voting_clf.predict(X_test)


print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(15, 8))
plt.title("Confusion Matrix")

sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Y Prediction")
plt.ylabel("Y True")
plt.show()

joblib.dump(voting_clf, "voting_clf_OnlineFraud")

new_model = joblib.load("voting_clf_OnlineFraud")
new_model.predict(X_test)

