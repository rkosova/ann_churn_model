import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("~/data/Churn_Modelling.csv")
X = dataset.iloc[: , 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_country = LabelEncoder()
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1]) 
labelencoder_X_gender = LabelEncoder()
X[:, 2] = labelencoder_X_gender.fit_transform(X[:, 2]) 
onehot = OneHotEncoder(categorical_features=[1])
X = onehot.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#making the ANN
import keras 
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#good way to choose layer neuron count is to use the average of input + output 
classifier.add(Dense(units = 6, input_dim = 11,kernel_initializer='uniform', activation = 'relu'))

#adding second layer
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid'))
#for 3+ dependent variables use softmax

#apply stochastic gradient descent (compiling)                                 criterion for improvment
classifier.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics = ['accuracy'])#categorical cross entropy for multiple inputs

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

#Making the predictions and evaluating model
y_pred = classifier.predict(X_test)
binary_pred = (y_pred > 0.5) # equivalent to [return 1 if y_pred > 0.5 else return 0]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, binary_pred)
