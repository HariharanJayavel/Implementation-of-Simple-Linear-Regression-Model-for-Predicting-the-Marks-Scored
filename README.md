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
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: HARIHARAN J
RegisterNumber: 212223240047
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='green')
plt.plot(x_train,reg.predict(x_train),color='red')
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='blue')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:

df.head()

![image](https://github.com/user-attachments/assets/11146019-6df3-40b4-93e3-c07b238f99d7)

df.tail()

![image](https://github.com/user-attachments/assets/4eb24b4f-1f8e-4be3-a2fc-12c7d35f4dcb)

Array values of X

![image](https://github.com/user-attachments/assets/709931e1-62fa-421a-8d70-931567ee383a)

Array values of Y

![image](https://github.com/user-attachments/assets/1e3326bb-85d4-4b86-a246-256e5a78be44)

Training Set Graph

![image](https://github.com/user-attachments/assets/d718e616-e56b-4280-b64b-dc0044befee8)

Testing Set Graph

![image](https://github.com/user-attachments/assets/b8981b82-f676-4895-97d6-e59461fbccde)

Values of MSE,MAE,RMSE

![image](https://github.com/user-attachments/assets/c0581472-ed43-483c-83e5-3fcaa2f9d44e)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
