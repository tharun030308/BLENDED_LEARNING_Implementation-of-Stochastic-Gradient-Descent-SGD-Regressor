# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the dataset.<br/>
2. Perform data preprocessing by removing unnecessary columns and converting categorical variables into numerical format.<br/>
3. Separate the dataset into input features (X) and target variable (y), and apply standardization.<br/>
4. Split the dataset into training and testing sets.<br/>
5. Create and train the SGD Regressor model using the training data.<br/>
6. Predict the output values using the test data.<br/>
7. Evaluate the model using performance metrics such as MSE, MAE, and R² score.<br/>
8. Visualize the results using a scatter plot of actual vs predicted values.

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

#Load Dataset
data=pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print(data.info())

#Data Preprocessing
#Dropping unnecessary columns and handling categorial variables
data=data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first=True)


#Splitting the data into features and target variable
x=data.drop('price',axis=1)
y=data['price']

#Standardizing the data
scaler=StandardScaler()
x=scaler.fit_transform(x)
y=scaler.fit_transform(np.array(y).reshape(-1,1))

#Splitting the dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Creating the SGD Regressor model
sgd_model=SGDRegressor(max_iter=1000,tol=1e-3)

#Fitting the model on the training data
sgd_model.fit(x_train,y_train)

#Making predictions
y_pred=sgd_model.predict(x_test)

#Evaluating model performance
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)


#Print Evaluation Metrics
print("Name:Barath B")
print("Reg no:25009091")
print("Mean Squared Error:",mse)
print("Mean Absolute Error:",mae)
print("R-squared Score:",r2)


#Print model coefficients
print("\nModel Coefficients")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)


#Visualizing actual vs predicted prices
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
plt.show()

print(y.shape)

```

## Output:
<img width="834" height="683" alt="image" src="https://github.com/user-attachments/assets/330a275c-c6ca-433a-b44e-715ef3631ee6" />
<img width="757" height="771" alt="image" src="https://github.com/user-attachments/assets/54d2e864-c90e-494b-bc76-ea254d23db47" />
<img width="868" height="217" alt="image" src="https://github.com/user-attachments/assets/c6dbd232-37a7-4fb9-8aaa-1ad6c1e0c67d" />
<img width="813" height="573" alt="image" src="https://github.com/user-attachments/assets/d42cb6ca-298b-47e5-a8e2-d3865f2b7bad" />



## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
