#info from http://scikit-learn.org/stable/modules/
#generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

#import sklearn statements
import sklearn as sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from sklearn.svm import LinearSVC

#for graph from http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import random_projection
from sklearn.metrics import accuracy_score

#other imports 
import scikitplot as skplt
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import validation_curve
from datetime import date

#Read data in using pandas
trainDataSet = pd.read_csv("happy.csv", sep = ',', header = None, low_memory = False)
#encode text data to integers using getDummies
traindata = pd.get_dummies(trainDataSet)
trainDataSet = trainDataSet[:250]
print("getting data")
# traindata = traindata.values

# Create decision Tree using major_category, month, year, to predict violent or not
# train split uses default gini node, split using train_test_split
X = trainDataSet.values[1:, 2:]
Y = trainDataSet.values[1:, 0]

cv = train_test_split(X, Y, test_size=.33, random_state= 20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state= 20)

model = LinearSVC()
t0= time.clock()


baseline = model.fit(X_train, Y_train)
Y_predict = baseline.predict(X_test)

baselinescore = accuracy_score(Y_test, Y_predict)
print(baselinescore)
t1= time.clock()
timetaken = str(t1-t0)
print("Computation Time" + timetaken)


#Finding the optimal component #

complist = [2,3,4,5,6]

##for each in complist:
##    comp = each
##    t0= time.clock()
##
##    print("Time started")
##    # Fit the PCA analysis
result = random_projection.GaussianRandomProjection(n_components=2).fit(traindata)
        
# print("Components"+ str(result.components_))
# result= result.fit(traindata)
# newdata= result.transform(traindata)
# lenght = len(newdata)
# print("Lenght of new data" + str(len(newdata)))
#
# newnewdata= []
# for each in range(lenght):
#     for every in range(each):
#         newnewdata.append(int(newdata[every]))
#
# X = newnewdata[1:, 1:]
# Y = newnewdata[1:,0]
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state= 20)
# print(X_train)
# print(Y_train)
# modelsfit = model.fit(X_train, Y_train)
# Y_predict = modelsfit.predict(X_test)
# scoreModel = accuracy_score(Y_test, Y_predict)
# print("Score" + str(scoreModel))
# t1= time.clock()
# timetaken = str(t1-t0)
# print("Computation Time" + timetaken)

# #RPvec= []
# Run RP with the optimal value 5 times, see what happens
# Graphical Representation of n = 3
# Fit the PCA analysis
newdata = result.transform(traindata)
traindata = traindata.values

print('Original vs. Transformed Data Graph')

plt.plot(traindata, 'o', markersize=7, color='blue', alpha=0.5)
plt.plot(traindata[0], 'o', markersize=7, color='blue', alpha=0.5, label = 'original')
plt.plot(newdata, '^', markersize=7, color='red', alpha=0.5)
plt.plot(newdata[0], '^', markersize=7, color='red', alpha=0.5, label='transformed')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-1,150])
plt.ylim([-10,10])
plt.legend()
plt.title('Original vs. Transformed Data Graph')

plt.show()