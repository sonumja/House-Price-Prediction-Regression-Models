# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('kc_house_data1.csv')
X = dataset.iloc[:, 2].values # Number of bedrooms
y = dataset.iloc[:, 20].values# Price of house
X=X.reshape(-1,1)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
#X_train=X_train.reshape(-1,1)
#y_train=y_train.reshape(-1,1)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)

# Visualising the Linear Regression results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, lin_reg.predict(X_train), color = 'blue')
plt.title('House Price vs Number of bedroom (Linear Regression)')
plt.xlabel('House Price')
plt.ylabel('Number of bedrooms')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, lin_reg_2.predict(poly_reg.fit_transform(X_train)), color = 'blue')
plt.title('House Price vs Number of bedroom (Polynomial Regression)')
plt.xlabel('House Price')
plt.ylabel('Number of bedrooms')
plt.show()

#Prediction
y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))

#Assessing the quality of fit
from sklearn.metrics import r2_score
print ('R-squared score for this model is ',r2_score(y_test,y_pred))
