import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from OnlineFraudDetection.OnlineFraudDetection_Pipeline import OnlineFraud_DataPrep

df = pd.read_csv("OnlineFraudDetection\onlineFraudDetection.csv")
X, y = OnlineFraud_DataPrep(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


new_model = joblib.load("OnlineFraudDetection/votingclf_onlineFraud_deneme")

testPredicted = new_model.predict(X_test)

print(classification_report(y_test, testPredicted))
