import pyro
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# reading data
path = "survey lung cancer.csv"
data = pd.read_csv(path)
# column to be ignored
columns = [i for i in data.columns]
ignore_columns = ["GENDER","AGE"] # for combination part

# return all combination of possible values
def return_permutation(col): # 0 or 1
    res = []
    com = []
    
    def backtrack(i = 0):
        if i>=len(col):
            res.append(tuple(com[::]))
            return
        com.append(0)
        backtrack(i+1)
        com.pop() 
        com.append(1)
        backtrack(i+1)   
        com.pop()
    backtrack()
    return res    

# non ignored columns
selected = []
for i in columns:
    if i not in ignore_columns:
        selected.append(i)

possibilities = return_permutation(selected) # taking the possibilities
# spliting data
Perc = 20 
Perc /= 100
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=Perc, random_state=42)    


# to count all possibilites
counter = {i:0 for i in possibilities}
# making counter
for i, row in data.iterrows():
    row_combinatio = []
    for col in selected:
        value = row[col]
        value = {1:0,2:1,"YES":1,"NO":0}[value]
        row_combinatio.append(value)
    counter[tuple(row_combinatio)] += 1

count_0 = 0
count_1 = 0
for k,v in counter.items():
    if v == 0:
        count_0 += 1 
    else:
        count_1 += 1    
print(count_0)        # 16251
print(count_1)
print(count_0+count_1)