# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries such as pandas, matplotlib, and scikit-learn's LinearRegression module.

2. Read the dataset (student_scores.csv) using pandas and extract the independent variable (Hours) and dependent variable (Scores).

3. Visualize the data using a scatter plot to understand the relationship between hours studied and marks scored.

4. Train the linear regression model using LinearRegression().fit(X, y), where X represents the number of hours studied and y represents the corresponding scores.

5. Make predictions using the trained model on the dataset and also predict for new values (e.g., 6.5 hours).

6. Visualize the regression line over the scatter plot to evaluate the model’s performance.

7. Evaluate the model by retrieving the slope, intercept, and predicted values.

8. Display the plot and the prediction results.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VINOTHKUMAR R
RegisterNumber:  212224040361
*/
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('/content/student_scores.csv')
df.head()
```

```
df.tail()

```

```
X = df.iloc[:,:-1].values
X
```

```
Y = df.iloc[:,1].values
Y
```

```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
Y_pred
```

```
Y_test
```

```
#graph
plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

```
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

```
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)

mae = mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

rmse = np.sqrt(mse)
print("RMSE = ",rmse)mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)

mae = mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

rmse = np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:

![1](https://github.com/user-attachments/assets/a7a710a4-590b-44f6-a19e-1955e3ac3344)
![2](https://github.com/user-attachments/assets/c3eb9d0d-2518-4b5a-a542-bbdb3f36d109)
![3](https://github.com/user-attachments/assets/1a968cc6-0440-41b2-9b6b-d4162bbc87c0)
![4](https://github.com/user-attachments/assets/79f0fc44-f085-4c85-aa20-a9917a2df172)
![5](https://github.com/user-attachments/assets/2ad8e1d9-0446-4923-9c46-216db197b4f6)

![6](https://github.com/user-attachments/assets/fe41b8c6-80b3-4ef1-b6f8-b9a25280b493)
![7](https://github.com/user-attachments/assets/f8fef0ac-c39c-48a9-88a4-20fc5dc84d1f)
![8](https://github.com/user-attachments/assets/59fe5479-200f-4fa4-b5a7-26aa080fe1bd)
![9](https://github.com/user-attachments/assets/22707441-9674-4662-93ea-0114fe2803c0)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
