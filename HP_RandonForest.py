#Importing the libraries

import pandas as pd
import numpy as np

#Importing the dataset
dataset=pd.read_csv('kc_house_data1.csv')
X=dataset.drop(['price','date'], axis=1).values
y=dataset.iloc[:,20].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split (X,y,test_size=0.2,random_state=0)

#Training the model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X_train,y_train)

#Making prediction for the test set
y_pred = regressor.predict(X_test)
y_pred.shape=(4323,1)

#Assessing the quality of fit
from sklearn.metrics import r2_score
print ('R-squared score for this model is ',r2_score(y_test, y_pred))

for x in range(len(y_pred)):
    print('Actual ',y_test[x],' Predicted ',y_pred[x])
