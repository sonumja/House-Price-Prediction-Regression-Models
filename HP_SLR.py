# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('kc_house_Data1.csv')
dataset['date']=dataset['date'].str[0:8] #Extract only the date from the date column
print(dataset['date'].dtype)
X = dataset.iloc[:,5].values #sqft_living is the X or feature
y = dataset.iloc[:, -1].values# Price is Y or label
X=X.reshape(-1,1)
y=y.reshape(-1,1)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
X_train=X_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)
y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')# This tells you where your line of best fit is so it takes X_train as attribute
plt.title('House Price vs Number of bedroom (Training set)')
plt.xlabel('House Price')
plt.ylabel('Number of bedrooms')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('House Price vs Number of bedroom (Test set)')
plt.xlabel('House Price')
plt.ylabel('Number of bedrooms')
plt.show()

# Visualising the predicted set results
plt.scatter(X_test, y_pred, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('House Price vs Number of bedroom (Predicted set)')
plt.xlabel('House Price')
plt.ylabel('Number of bedrooms')
plt.show()

# Check the Rsquared value
#Assessing the quality of fit
from sklearn.metrics import r2_score
print ('R-squared score for this model is ',r2_score(y_test, y_pred))