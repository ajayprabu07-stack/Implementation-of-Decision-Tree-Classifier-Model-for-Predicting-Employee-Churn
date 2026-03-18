# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas

2. Import Decision tree classifier

3. Fit the data in the model

4. Find the accuracy score

## Program:

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: A.Ajayprabu
RegisterNumber:212225220005
*/
import pandas as pd

# Load dataset
data = pd.read_csv("Employee.csv")

print("data.head():")
print(data.head())

print("data.info():")
print(data.info())

print("isnull() and sum():")
print(data.isnull().sum())

print("data value counts():")
print(data["left"].value_counts())

# Encoding salary column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

print("data.head() for Salary:")
data["salary"] = le.fit_transform(data["salary"])
print(data.head())

# Selecting features
X = data[["satisfaction_level", "last_evaluation", "number_project",
          "average_montly_hours", "time_spend_company",
          "Work_accident", "promotion_last_5years", "salary"]]

print("X.head():")
print(X.head())

# Target variable
y = data["left"]

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")

dt.fit(X_train, y_train)

# Prediction
y_pred = dt.predict(X_test)

# Accuracy
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy value:", accuracy)

# Data prediction example
print("Data Prediction:")
print(dt.predict([[0.5, 0.8, 3, 160, 3, 0, 0, 1]]))

# Plot Decision Tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=X.columns, class_names=['Stay','Left'], filled=True)
plt.show()

## Output:
<img width="672" height="856" alt="Screenshot 2026-03-18 151919" src="https://github.com/user-attachments/assets/7e2bb405-e074-42c2-a270-4f472b3504c8" />
<img width="1022" height="891" alt="Screenshot 2026-03-18 151937" src="https://github.com/user-attachments/assets/975f8312-5ad3-4012-9308-3c7a187891d3" />
<img width="1004" height="384" alt="Screenshot 2026-03-18 152322" src="https://github.com/user-attachments/assets/4d0dda24-4684-4a6b-99c7-72e236b9b449" />
<img width="687" height="443" alt="Screenshot 2026-03-18 152532" src="https://github.com/user-attachments/assets/70344279-d0d7-4da8-a2bd-1afd52002cd2" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
