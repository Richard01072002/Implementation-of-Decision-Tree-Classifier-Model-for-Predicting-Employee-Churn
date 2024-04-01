# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

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
Developed by: 
RegisterNumber:  
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
![decision tree classifier model](sam.png)

<img width="456" alt="Screenshot 2024-04-01 at 9 06 30 AM" src="https://github.com/AkilaMohan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141472248/52a35d1d-c4fb-43c6-aa18-539f85f37384">

<img width="219" alt="Screenshot 2024-04-01 at 9 06 38 AM" src="https://github.com/AkilaMohan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141472248/c4274b6d-f887-4bdb-87e6-45ca735e0861">

<img width="1216" alt="Screenshot 2024-04-01 at 9 07 04 AM" src="https://github.com/AkilaMohan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141472248/c3330098-3ad0-4a63-bd20-c642f16652c6">

<img width="1200" alt="Screenshot 2024-04-01 at 9 07 16 AM" src="https://github.com/AkilaMohan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141472248/bdbc9980-03d8-465a-a665-4321c95a0364">

<img width="232" alt="Screenshot 2024-04-01 at 9 07 34 AM" src="https://github.com/AkilaMohan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141472248/46cf0a9d-d370-40a8-8e01-a9c55984e6c3">

<img width="1205" alt="Screenshot 2024-04-01 at 9 07 54 AM" src="https://github.com/AkilaMohan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/141472248/69d9046a-ea30-4c34-bf81-33dddf7ccda2">


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
