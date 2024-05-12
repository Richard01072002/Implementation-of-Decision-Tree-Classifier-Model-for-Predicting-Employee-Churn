# EXP-6 Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: RICGARDSON A
RegisterNumber:  212222233005
*/

import pandas as pd
data = pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
```
data.head()
data.info()
```
<img width="456" alt="Screenshot 2024-04-01 at 9 06 30 AM" src="https://github.com/Richard01072002/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141472248/4bd7172f-f938-4251-a26d-b1a40c6f81f1">

```
data.isnull().sum()
data["left"].value_counts()
```
<img width="219" alt="Screenshot 2024-04-01 at 9 06 38 AM" src="https://github.com/Richard01072002/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141472248/66e82564-2d50-4bc7-bf40-217f98a844e1">

```
data["salary"] = le.fit_transform(data["salary"])
data.head()
```
<img width="1216" alt="Screenshot 2024-04-01 at 9 07 04 AM" src="https://github.com/Richard01072002/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141472248/35f7d11d-796c-4291-b6f0-b68347d36075">

```
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
<img width="1200" alt="Screenshot 2024-04-01 at 9 07 16 AM" src="https://github.com/Richard01072002/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141472248/b4881738-03aa-4cf3-b7f6-73511c509a57">

```
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```
<img width="232" alt="Screenshot 2024-04-01 at 9 07 34 AM" src="https://github.com/Richard01072002/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141472248/6c4c61d6-03f3-43f3-b3fe-0f205bd570a2">


```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
<img width="1205" alt="Screenshot 2024-04-01 at 9 07 54 AM" src="https://github.com/Richard01072002/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141472248/7735c8c0-b001-4104-a338-2d351e575298">


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
