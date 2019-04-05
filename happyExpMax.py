#Sheena Ganju, CS 4641 HW 1
#Decision Trees

#info from http://scikit-learn.org/stable/modules/
#generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

#import sklearn statements
import sklearn as sklearn
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

#for graph from http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#k-means specific imports 
from sklearn import mixture
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
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist, pdist
from sklearn.mixture import GaussianMixture

#Read data in using pandas
trainDataSet = pd.read_csv("happy.csv", sep = ',', header = None, low_memory = False)


t0 = time.clock()
#encode text data to integers using getDummies
traindata = pd.get_dummies(trainDataSet)
# Create decision Tree using major_category, month, year, to predict violent or not 
# train split uses default gini node, split using train_test_split

X = trainDataSet.values[1:, 2:]
Y = trainDataSet.values[1:, 0]

gmm = mixture.GaussianMixture(n_components=1, covariance_type='full')
gmm.fit(traindata)


# print(gmm.means_)
# print(gmm.lower_bound_)

# plot_results(traindata, gmm.predict(traindata), gmm.means_, gmm.covariances_, 0,
#             'Gaussian Mixture')

cv = train_test_split(X, Y, test_size=.30, random_state= 20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30, random_state= 20)
#
# print ("---------------")
# print (X_train.size)
# print (X_test.size)
# print ("---------------")
# print (X_train.shape)
# print (X_test.shape)
# print (X_test.reshape(-1, 1).shape)
# print ("---------------")


# X_test.reshape(-1, 1)
#
# Y_prediction = gmm.predict(X_test)
#
# from sklearn.metrics import log_loss
# loss = log_loss(Y_test, Y_prediction)*100
# print("Loss: " + str(loss))


t1= time.clock()
time = str(t1-t0)
print("Computation Time: " + time)


skplt.estimators.plot_learning_curve(gmm, X, Y, title = "Learning Curve: GMM")
plt.show()




WINE_DATA_FILE = 'happy.csv'
wine_data = pd.read_csv(WINE_DATA_FILE)
#wine_data.reshape(-1,1)
wine_x = wine_data[[x for x in wine_data.columns if x != '"Binary Happiness"']]
K = range(1, 40)
GMM_c = [GaussianMixture(n_components=k).fit(traindata) for k in K]
GMM_w = [GaussianMixture(n_components=k).fit(wine_x) for k in K]
print("Trained EM models")
# LL_c = [gmm.score(char_x) for gmm in GMM_c]
LL_w = [gmm.score(wine_x) for gmm in GMM_w]
print("Calculated the log likelihood for each k")
# BIC_c = [gmm.bic(char_x) for gmm in GMM_c]
BIC_w = [gmm.bic(wine_x) for gmm in GMM_w]
print("Calculated the BICs for each K")
# AIC_c = [gmm.aic(char_x) for gmm in GMM_c]
AIC_w = [gmm.aic(wine_x) for gmm in GMM_w]
print("Calculated the AICs for each K")
plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.plot(K, BIC_c, '*-', label='BIC for Letter Recognition')
# ax.plot(K, AIC_c, '*-', label='AIC for Letter Recognition')
ax.plot(K, BIC_w, '*-', label='BIC for Happiness')
ax.plot(K, AIC_w, '*-', label='AIC for Happiness')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Inference scores')
plt.legend(loc='best')
plt.title('Bayesian and Akaike Information Criterion Curve')
fig.savefig('graphs/em/bic_aic.png')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.plot(K, LL_c, '*-', label='Letter Recognition')
ax.plot(K, LL_w, '*-', label='Happiness')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood Curve')
plt.legend(loc='best')
fig.savefig('graphs/em/log_likelihood.png')
plt.show()
