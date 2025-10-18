import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# Parameters

PATH = "survey lung cancer.csv"
ignored = []

# reading and filtering data
data = pd.read_csv(PATH)
data = data.drop(columns=ignored)
target_col = "LUNG_CANCER"
Perc = 40
Perc /= 100
data["GENDER"] = data["GENDER"].apply(lambda x: 0 if x=="M" else 1)

# data part
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=Perc, random_state=42)    

# model 
model = LogisticRegression()
model = model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(classification_report(y_test,y_pred))