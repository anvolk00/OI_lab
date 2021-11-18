## import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import warnings
warnings.filterwarnings("ignore")

## import data
mat= loadmat('digits.mat')
X = mat['X']
y = mat['y'].ravel()
# make 0 to 9
y = y%10  

## normalization - Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
Xnorm = sc_X.fit_transform(X)

## split train and test data
## shuffling applied to the data before applying the split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xnorm, y, test_size = 0.25, random_state = 1) 

## print summary on test/train split
print ('')
print ('X_train dimensions are ', X_train.shape, '// y_train dimensions are ', y_train.shape)
print ('X_test dimensions are ', X_test.shape, '// y_test dimensions are ', y_test.shape)
print ('')

## import MLPClassifier
from sklearn.neural_network import MLPClassifier

## in all cases alpha is default 

## set parameters for 1-Layer Neural Network model using 'relu'
clf1= MLPClassifier(
    hidden_layer_sizes = (100),
    solver = 'adam',
    activation = 'relu', 
    max_iter = 500,
)

## fit 1-Layer Neural Network model using 'relu' and print accuracy results
clf1.fit(X_train,y_train)
print("train accuracy for 1-Layer Neural Network with 100 neurons using 'relu' = {:.3%}".format(clf1.score (X_train, y_train)))
print("test accuracy for 1-Layer Neural Network with 100 neurons using 'relu' = {:.3%}".format(clf1.score (X_test, y_test)))
print ('')

##---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## set parameters for 2-Layers Neural Network model (3,3) using 'identity'
clf2= MLPClassifier(
    hidden_layer_sizes = (3,3),
    solver = 'adam',
    activation = 'identity', 
    max_iter = 500,
)

## fit 2-Layers Neural Network model (3,3) using 'identity' and print accuracy results
clf2.fit(X_train,y_train)
print("train accuracy for 2-Layers Neural Network (3,3) using 'identity' = {:.3%}".format(clf2.score (X_train, y_train)))
print("test accuracy for 2-Layers Neural Network (3,3) using 'identity' = {:.3%}".format(clf2.score (X_test, y_test)))
print ('')

## set parameters for 2-Layers Neural Network model (3,3) using 'relu'
clf3= MLPClassifier(
    hidden_layer_sizes = (3,3),
    solver = 'adam',
    activation = 'relu', 
    max_iter = 500,
)

## fit 2-Layers Neural Network model (3,3) using 'relu' and print accuracy results
clf3.fit(X_train,y_train)
print("train accuracy for 2-Layers Neural Network (3,3) using 'relu' = {:.3%}".format(clf3.score (X_train, y_train)))
print("test accuracy for 2-Layers Neural Network (3,3) using 'relu' = {:.3%}".format(clf3.score (X_test, y_test)))
print ('')

##---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## set parameters for 3-Layers Neural Network model (20,7,10) using 'identity'
clf4= MLPClassifier(
    hidden_layer_sizes = (20,7,10),
    solver = 'adam',
    activation = 'identity', 
    max_iter = 500,
)

## fit 3-Layers Neural Network model (20,7,10) using 'identity' and print accuracy results
clf4.fit(X_train,y_train)
print("train accuracy for 3-Layers Neural Network (20,7,10) using 'identity' = {:.3%}".format(clf4.score (X_train, y_train)))
print("test accuracy for 3-Layers Neural Network (20,7,10) using 'identity' = {:.3%}".format(clf4.score (X_test, y_test)))
print ('')

## set parameters for 3-Layers Neural Network model (20,7,10) using 'relu'
clf5= MLPClassifier(
    hidden_layer_sizes = (20,7,10),
    solver = 'adam',
    activation = 'relu', 
    max_iter = 500,
)

## fit 3-Layers Neural Network model (20,7,10) using 'relu' and print accuracy results
clf5.fit(X_train,y_train)
print("train accuracy for 3-Layers Neural Network (20,7,10) using 'relu' = {:.3%}".format(clf5.score (X_train, y_train)))
print("test accuracy for 3-Layers Neural Network (20,7,10) using 'relu' = {:.3%}".format(clf5.score (X_test, y_test)))
print ('')

## set parameters for 3-Layers Neural Network model (20,7,10) using 'tanh'
clf6= MLPClassifier(
    hidden_layer_sizes = (20,7,10),
    solver = 'adam',
    activation = 'tanh', 
    max_iter = 500,
)

## fit 3-Layers Neural Network model (20,7,10) using 'tanh' and print accuracy results
clf6.fit(X_train,y_train)
print("train accuracy for 3-Layers Neural Network (20,7,10) using 'tanh' = {:.3%}".format(clf6.score (X_train, y_train)))
print("test accuracy for 3-Layers Neural Network (20,7,10) using 'tanh' = {:.3%}".format(clf6.score (X_test, y_test)))
print ('')

##---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## import SVC for SupportVectorMachine (SVM) method
from sklearn.svm import SVC

## fit model using SVC
clf7 = SVC(C=10, kernel='rbf', gamma=0.001).fit(X_train, y_train)
print("train accuracy for SVM method = {:.3%}".format(clf7.score (X_train, y_train)))
print("test accuracy for SVM method = {:.3%}".format(clf7.score (X_test, y_test)))

##---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## func for visualization
def draw_pixels_img(x, ax = None, title=None):
    '''
    :param x: ndarray - row
    '''
    img_width = int(np.sqrt(x.shape[0]))
    #img_height = x.shape[0]/img_width
    try:
        data = x.reshape(img_width, -1).T
    except:
        SystemExit('Cannot compute the size of the picture')
    if ax:
        plt.sca(ax)
    else:
        plt.figure(figsize=(2, 2))
    plt.imshow(data, cmap='Greys', interpolation='nearest')
    plt.axis('off')
    if not title is None:
        plt.title(title)
        
def draw_digits_samples(X,n_rows= 10, n_cols = 10, y=None):
    indices = np.random.randint(0, len(X), n_rows*n_cols)
    for i in range (n_rows): 
        for j in range (n_cols):
            index = n_rows*i+j           
            ax = plt.subplot(n_rows,n_cols,index+1) 
            if y is None: 
                draw_pixels_img(X[indices[index]], ax)
            else:
                draw_pixels_img(X[indices[index]], ax, title=y[indices[index]])
    plt.tight_layout(h_pad=-1) 

## visualize results for clf1
plt.figure().text(.05,.05,"1-Layer Neural Network with 100 neurons using 'relu'")
predicted= clf1.predict(X_test)
draw_digits_samples(X_test, n_rows= 4, n_cols = 6, y = predicted)

## visualize results for clf2
plt.figure().text(.05,.05,"2-Layers Neural Network model (3,3) using 'identity'")
predicted= clf2.predict(X_test)
draw_digits_samples(X_test, n_rows= 4, n_cols = 6, y = predicted)

## visualize results for clf3
plt.figure().text(.05,.05,"2-Layers Neural Network model (3,3) using 'relu'")
predicted= clf3.predict(X_test)
draw_digits_samples(X_test, n_rows= 4, n_cols = 6, y = predicted)

## visualize results for clf4
plt.figure().text(.05,.05,"3-Layers Neural Network model (20,7,10) using 'identity'")
predicted= clf4.predict(X_test)
draw_digits_samples(X_test, n_rows= 4, n_cols = 6, y = predicted)

## visualize results for clf5
plt.figure().text(.05,.05,"3-Layers Neural Network model (20,7,10) using 'relu'")
predicted= clf5.predict(X_test)
draw_digits_samples(X_test, n_rows= 4, n_cols = 6, y = predicted)

## visualize results for clf6
plt.figure().text(.05,.05,"3-Layers Neural Network model (20,7,10) using 'tanh'")
predicted= clf6.predict(X_test)
draw_digits_samples(X_test, n_rows= 4, n_cols = 6, y = predicted)

## visualize results for clf7
plt.figure().text(.05,.05,"SVM method")
predicted= clf7.predict(X_test)
draw_digits_samples(X_test, n_rows= 4, n_cols = 6, y = predicted)
