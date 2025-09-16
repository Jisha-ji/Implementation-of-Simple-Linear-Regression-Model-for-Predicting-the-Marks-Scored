# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: JISHA BOSSNE SJ
RegisterNumber:  212224230106

```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)

print(df.head())
print(df.tail())

x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

DATASET

<img width="195" height="521" alt="ml 1" src="https://github.com/user-attachments/assets/0b0c92a4-5989-4aa0-ad1c-9f6e231181ce" />


HEAD VALUES

<img width="162" height="117" alt="ml 2" src="https://github.com/user-attachments/assets/7af0e5b4-ab13-4b50-8034-c816fff6af9a" />


TAIL VALUES

<img width="160" height="117" alt="ml 3" src="https://github.com/user-attachments/assets/fe142b89-9c6f-4006-aa74-da79ecff145e" />


X AND Y VALUES

<img width="610" height="517" alt="ml 4" src="https://github.com/user-attachments/assets/81614ca8-8f4d-4f94-b53f-0862a47822cc" />


PREDICTED VALUES OF X AND Y

<img width="666" height="67" alt="ml 5" src="https://github.com/user-attachments/assets/2dd2e619-5e87-4c27-a93b-facbb35925b6" />


TRAINING SET

<img width="728" height="518" alt="ml 6" src="https://github.com/user-attachments/assets/7755f1c8-2f27-477f-95d1-cd503929117d" />


TESTING SET AND MSE,MAE and RMSE

<img width="727" height="612" alt="ml 7" src="https://github.com/user-attachments/assets/3bc8cd7b-4b29-4a7f-ae23-e1ada94c633b" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
