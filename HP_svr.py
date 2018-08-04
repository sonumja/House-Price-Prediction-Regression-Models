# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('kc_house_data1.csv')
X = dataset.iloc[:, 3:19].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

# Predicting a new result
X_test = sc_X.fit_transform(X_test)
y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)

#Assessing the quality of fit
from sklearn.metrics import r2_score
print ('R-squared score for this model is ',r2_score(y_test,y_pred))