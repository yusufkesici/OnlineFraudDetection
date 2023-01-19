import pandas as pd
import numpy as np
from helpers.eda import *
from helpers.data_prep import *
import seaborn as sns
import matplotlib.pyplot as plt
import random

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

df = pd.read_csv("OnlineFraudDetection/onlineFraudDetection.csv")
df.head()
load_df = df.copy()

############################
# EXPLORATORY DATA ANALYZE
############################


check_df(df)

df["isFraud"].value_counts(normalize=True)
df["isFraud"].value_counts()

df.duplicated().any()

fraud_no = df[df["isFraud"] == 0].sample(8213, random_state=42)
fraud_yes = df[df["isFraud"] == 1]

df = pd.concat([fraud_yes, fraud_no])

df["type"].value_counts()
df["isFraud"].value_counts()

df[df["isFraud"] == 1][["type"]].value_counts()
df[df["isFraud"] == 0][["type"]].value_counts()

# Fraud olanlarların type ları cash_out veya transfer oluyor.


df["isFlaggedFraud"].value_counts()
df.drop("isFlaggedFraud", axis=1, inplace=True)

sns.pairplot(df)

sns.scatterplot(x="oldbalanceOrg", y="newbalanceOrig", hue="isFraud", data=df)
sns.scatterplot(x="oldbalanceDest", y="oldbalanceDest", hue="isFraud", data=df)

sns.scatterplot(x="oldbalanceOrg", y="newbalanceOrig", hue="type", data=df)
sns.scatterplot(x="oldbalanceDest", y="oldbalanceDest", hue="type", data=df)

df["nameDest"].nunique()
df["nameOrig"].nunique()

df.drop(["nameDest", "nameOrig"], axis=1, inplace=True)

########################################
# Kategorik ve sayısal değişken analizi
########################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Kategorik değişken analizi
for col in cat_cols:
    cat_summary(df, col, True)
    plt.show(block=True)

# Sayısal değişken analizi
for col in num_cols:
    num_summary(df, col, True)
    plt.show(block=True)

# Değişken çaprazlama görselleştirme
sns.pairplot(df, hue="isFraud", size=5)

# Kategorik değişkenlerin Target değişken ile analizi
for col in cat_cols:
    target_summary_with_cat(df, "isFraud", col)

# Sayısal değişkenlerin Target değişken ile analizi
for col in num_cols:
    target_summary_with_num(df, "isFraud", col)

# fraud olanların step ortalaması yaklaşık 120 daha fazla
# fraud olanların amount ortalaması olmayanların yaklaşık 8 kat daha fazla
# fraud olanların oldbalanceOrg ortalaması olmayanların 2 kat daha fazla
# fraud olanların newbalanceOrig ortalaması olmayanların yaklaşık 1/4 kadarı
# fraud olanların oldbalanceDest ortalaması olmayanların yaklaşık 1/2 kadarı


# Aykırı değer analizi
for col in num_cols:
    print(col, check_outlier(df, col))  # 0.01 0.99 threshold

# step False
# amount True
# oldbalanceOrg True
# newbalanceOrig True
# oldbalanceDest True
# newbalanceDest True


drop_index = grab_outliers(df, "amount", True)
df.drop(drop_index, inplace=True)

for col in num_cols:
    print(col, outlier_thresholds(df, col))

grab_outliers(df, "oldbalanceOrg", True)
grab_outliers(df, "newbalanceOrig", True)
grab_outliers(df, "oldbalanceDest", True)
grab_outliers(df, "newbalanceDest", True)

# Eksik değer analizi

df.isna().sum()

# korelasyon analizi

corr = df.corr()

plt.figure(figsize=(15, 8))
sns.heatmap(corr, annot=True)
plt.show()

# target variable correleated with amount and step a little ~0.30
# oldbalanceOrg and amount correaleted ~0.60
# oldbalanceOrg and newbalanceOrg correaleted ~0.80
# oldbalanceDest and newbalanceDest correaleted ~0.90

scatterplot_cols = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "amount", "step"]

for i in scatterplot_cols:
    for j in scatterplot_cols:
        if j != i:
            plt.figure(figsize=(15, 8))
            sns.scatterplot(x=i, y=j, data=df, hue="isFraud")
            plt.show(block=True)
        else:
            plt.figure(figsize=(15, 8))
            sns.kdeplot(data=df, x=i, hue="isFraud", shade="soft")
            plt.show(block=True)

for i in scatterplot_cols:
    for j in scatterplot_cols:
        if j != i:
            plt.figure(figsize=(15, 8))
            sns.scatterplot(x=i, y=j, data=df, hue="type")
            plt.show(block=True)
        else:
            plt.figure(figsize=(15, 8))
            sns.kdeplot(data=df, x=i, hue="type", shade="soft")
            plt.show(block=True)

dff = df.copy()
df = dff.copy()

###############
# BASE MODEL
###############

df = one_hot_encoder(df, ["type"], True)

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# FEATURE ENGİNEERİNG
df = dff.copy()

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


##################################
# MODEL WİTH FEATURE ENGİNEERİNG
##################################

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=32)

from imblearn.under_sampling import RandomUnderSampler
# transform the dataset
ranUnSample = RandomUnderSampler()
X_ranUnSample, y_ranUnSample = ranUnSample.fit_resample(X_train, y_train)


RidgeClassifier_params = {'alpha': [1.0, 20, 50, 100, 200]}

logistic_params = {'max_iter': [100, 200, 500],
                   "solver": ['liblinear']
                   }

GBM_params = {"n_estimators": [5, 50, 100, 250, 500],
              "max_depth": [1, 3, 5, 7, 9],
              "learning_rate": [0.01, 0.1, 1, 10]}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "sqrt"],
             "min_samples_split": [2, 15, 20],
             "n_estimators": [200, 300]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "subsample": [1.0, 0.9, 0.5, 0.3]}

regressors = [('LR', LogisticRegression(), logistic_params),
              ('Ridge', RidgeClassifier(), RidgeClassifier_params),
              ('GBM', GradientBoostingClassifier(), GBM_params),
              ('LGBM', LGBMClassifier(), lightgbm_params),
              ('RF', RandomForestClassifier(), rf_params),
              ]

best_models = {}
scoring = ['precision', 'recall', 'f1', "accuracy"]
for name, regressor, params in regressors:
    print(f"########## {name} ##########")

    cv_results = cross_validate(final_model, X_ranUnSample, y_ranUnSample, cv=3, scoring=scoring)
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X_ranUnSample, y_ranUnSample)
    final_model = regressor.set_params(**gs_best.best_params_)
    print(f"########## After GridSearchCV Final Results ##########")

    cv_results = cross_validate(final_model, X_ranUnSample, y_ranUnSample, cv=3, scoring=scoring)
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")
    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
    best_models[name] = final_model

# 'LR': LogisticRegression(solver='liblinear'),
# 'Ridge': RidgeClassifier(),
# 'GBM': GradientBoostingClassifier(max_depth=5, n_estimators=250),
# 'LGBM': LGBMClassifier(n_estimators=500),
# 'RF': RandomForestClassifier(max_depth=15, n_estimators=200)


final_model = LGBMClassifier(n_estimators=500)

final_model.fit(X_ranUnSample, y_ranUnSample)

y_pred = final_model.predict(X_test)

accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(15, 8))
plt.title("Confusion Matrix")

sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Y Prediction")
plt.ylabel("Y True")
plt.show()



######################
# FEATURE IMPORTANCES
######################

feature_importance = pd.DataFrame({"Feature": X.columns, "Value": final_model.feature_importances_})
plt.figure(figsize=(12, 12))
sns.set(font_scale=1)
sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False)[:10])

plt.title('Features')
plt.tight_layout()
plt.show()
