## import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## import dataset
dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:, 0:5].values
y = dataset.iloc[:, 5].values

## Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

## normalization - Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
Xnorm = sc_X.fit_transform(X)

## split train and test data
## shuffling applied to the data before applying the split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xnorm, y, test_size = 0.25, random_state = 7) 

## Fitting classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
## set K
k=40 
## creating KNN model
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(X_train, y_train)

##Predicting
y_pred = classifier.predict(X_test)

## Predicting and training for K value from 0 to 60
from sklearn import metrics
error_rate = []
for i in range(1,60):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(X_train,y_train)
 i_pred = knn.predict(X_test)
 print("Accuracy of model at K=", i,"is ", metrics.accuracy_score(y_test, i_pred))
 error_rate.append(np.mean(i_pred != y_test))

## plot **Error rate** to **K value**
plt.figure(figsize=(10,6))
plt.plot(range(1,60),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.grid(axis='x', color='0.5')
plt.grid(axis='y', color='0.5')
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
## print best K
print("Minimum error:",min(error_rate),"at K =",error_rate.index(min(error_rate)), "- best K for KNN")







