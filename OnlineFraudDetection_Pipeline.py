import pandas as pd
import numpy as np
import joblib
from helpers.eda import *
from helpers.data_prep import *
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

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

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    return X, y


rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "sqrt"],
             "min_samples_split": [2, 15, 20],
             "n_estimators": [100, 200, 300]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 1000, 10000],
                   "subsample": [1.0, 0.5]}

CART_params = {'max_depth': [2, 3, 5, 10, 20],
               'min_samples_leaf': [5, 10, 20, 50, 100],
               'criterion': ["gini", "entropy"]}

regressors = [("CART", DecisionTreeClassifier(), CART_params),
              ('LGBM', LGBMClassifier(), lightgbm_params),
              ('RF', RandomForestClassifier(), rf_params),
              ]

scoring = ['precision', 'recall', 'f1', "accuracy"]


def hyperparameter_optimization(X, y, cv=3):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        cv_results = cross_validate(regressor, X, y, cv=cv, scoring=scoring)
        print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
        print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
        print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
        print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_)
        print(f"########## After GridSearchCV Final Results ##########", end="\n")

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
        print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
        print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
        print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


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


def main():
    df = pd.read_csv("..\OnlineFraudDetection\onlineFraudDetection.csv")
    X, y = OnlineFraud_DataPrep(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    ranUnSample = RandomUnderSampler()

    X_ranUnSample, y_ranUnSample = ranUnSample.fit_resample(X_train, y_train)

    best_models = hyperparameter_optimization(X_ranUnSample, y_ranUnSample)
    voting_clf = voting_classifier(best_models, X_ranUnSample, y_ranUnSample)
    joblib.dump(voting_clf, "votingclf_onlineFraud_deneme")
    return voting_clf


if __name__ == "__main__":
    print("Processing...")
    main()
