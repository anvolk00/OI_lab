## import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## create synthetic dataset 
from sklearn.datasets import make_regression
X_R1, y_R1 = make_regression(
n_samples = 1000,
n_features=1,
n_informative=1,
bias = 0,
noise = 10) 

## normalization - Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_R1 = sc_X.fit_transform(X_R1)

## split train and test data
## shuffling applied to the data before applying the split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, test_size = 0.2, random_state = 3)

## Fitting regressor to the training set
from sklearn.neighbors import KNeighborsRegressor
## set K
k=8 
## creating KNN-regressor model
knn_reg = KNeighborsRegressor(n_neighbors = k)
knn_reg.fit(X_train, y_train)

## R2 score
(knn_reg.score(X_test, y_test))  
## or this way
## R2 score from  sklearn.metrics
z= knn_reg.predict(X_test)
from sklearn.metrics import r2_score
(r2_score(y_test,  z))

## regression R2 score for K from 1 to 100
from sklearn import metrics
scp=[]
for i in range(1,100):
 knn_reg = KNeighborsRegressor(n_neighbors=i)
 knn_reg.fit(X_train, y_train)
 z= knn_reg.predict(X_test)
 print("R2 score of model at K=", i,"is ", r2_score(y_test,  z))
 scp.append(metrics.r2_score(y_test, z))

## plot regression KNN
X_test = np.linspace(-3, 3, 200).reshape(-1,1)
plt.grid(axis='x', color='0.5')
plt.grid(axis='y', color='0.5')
plt.plot(X_train, y_train, 'o', label='True Value', alpha=0.8)
plt.plot(X_test, knn_reg.predict(X_test), '-', markersize = 10, label='Predicted', alpha=0.8)
plt.xlabel('Input feature')
plt.ylabel('Target value')
plt.title('KNN regression (K={})'.format(scp.index(max(scp))))
plt.legend()

## plot **R2 Score** to **K value**
plt.figure(figsize=(10,6))
plt.grid(axis='x', color='0.5')
plt.grid(axis='y', color='0.5')
plt.plot(range(1,100),scp,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('R2 Score vs. K Value')
plt.xlabel('K')
plt.ylabel('R2 Score')

## print best R2 score for K 
print("Best R2 score:",max(scp),"at K =",scp.index(max(scp)), "- best K for KNN-regressor")
