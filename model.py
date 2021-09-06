import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('DataOfSongs.csv')
print(df)

#X has independent variables
X = df.iloc[:,:11]
#y has dependent variable
y = df.hit

#plit data into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# Implement Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Check Accuracy
score = classifier.score(X_test, y_test)
print("Accuracy = ",score)

# Create a Pickle file 
pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()

# Predict a value
predicted_value = classifier.predict([[0.346, 0.660,4, 0.0332,0.01720, 0.000038, 0.0550, 0.600, 92.990, 279740, -10.276]])
print("Predicted Value =",predicted_value)