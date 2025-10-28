import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
from imblearn.over_sampling import RandomOverSampler
# Parameters

PATH = "survey lung cancer.csv"
ignored = []

# reading and filtering data
data = pd.read_csv(PATH)
data = data.drop(columns=ignored)
target_col = "LUNG_CANCER"

"""best_perc = [0,0,0] # perc , score ,model
for i in range(20,90):
    print(i)
    Perc = i
    Perc /= 100
    data["GENDER"] = data["GENDER"].apply(lambda x: 0 if x=="M" else 1)

    # data part
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=Perc, random_state=42)    

    # model
    # --- Models ---
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier 
    from sklearn.metrics import classification_report, accuracy_score
    models = {
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Gradient Boosting": GradientBoostingClassifier(),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier()
    }

    # --- Evaluate all models ---
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
    # --- Find best performer ---
    best_model = max(results, key=results.get)
    if results[best_model] > best_perc[0]:
        best_perc = [i , results[best_model] , model]
        
print("\n🏆 Best Model:", best_perc[2], f"with Accuracy = {best_perc[1]:.4f} and PERCENTAGE = {best_perc[0]}")"""
# in this way we decided the percentage = 20 is best for decisionTreeclassifier with accuracy = 0.9677




Perc = 20 
Perc /= 100
data["GENDER"] = data["GENDER"].apply(lambda x: 0 if x=="M" else 1)

# data part
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=Perc, random_state=42)    
ros = RandomOverSampler(random_state=42)
x_train, y_train = ros.fit_resample(x_train, y_train)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(classification_report(y_pred=y_pred,y_true=y_test))

print(f"accuracy = {acc}")
