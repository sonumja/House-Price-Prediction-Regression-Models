# Decision Tree Regression

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('kc_house_data.csv')
X = dataset.iloc[:, 2:18].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

for x in range(len(y_pred)):
    print('Actual ',y_test[x],' Predicted ',y_pred[x])
    
#Assessing the quality of fit
from sklearn.metrics import r2_score
print ('R-squared score for this model is ',r2_score(y_test,y_pred))