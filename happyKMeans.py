#Decision Trees
from __future__ import print_function

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
print(__doc__)

#k-means specific imports 
from sklearn.cluster import KMeans
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
# Create decision Tree using major_category, month, year, to predict violent or not 
# train split uses default gini node, split using train_test_split

X = trainDataSet.values[1:, 2:]
Y = trainDataSet.values[1:, 0:1]

cv = train_test_split(X, Y, test_size=.33, random_state= 20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state= 20)


# clf = KMeans(n_clusters=2, random_state=10)

# clf = clf.fit(X_train, Y_train)

# train_prediction = clf.predict(X_train)
# trainaccuracy = accuracy_score(train_prediction, Y_train)*100
# print("The training accuracy for this is " +str(trainaccuracy))
# #output
# Y_prediction = clf.predict(X_test)
# accuracy = accuracy_score(Y_test, Y_prediction)*100
# print("The test classification works with " + str(accuracy) + "% accuracy without pruning")
# skplt.estimators.plot_learning_curve(clf, X, Y, title = "Happy K Means Cluster Learning Curve")




# cluster_labels = clusterer.fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=cluster_labels)
# plt.title("Happy K-Means Clustering")
# plt.show()

#start timer
t0= time.clock()

#cv = train_test_split(X, Y, test_size=.30, random_state= 20)
##X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state= 20)
##

k = 2
# clf = sklearn.cluster.KMeans(n_clusters= k, init='k-means++', n_init = 5, max_iter = 500)

# clf = clf.fit(traindata)

range_n_clusters = range(2,30)
y=Y
silhouetteScore = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    silhouetteScore.append(silhouette_avg * 100)

fig = plt.figure()
plt.plot(range_n_clusters, silhouetteScore, 'r', label="SilhouetteScore Average")
plt.xlabel('K Clusters')
plt.ylabel('Within Cluster SSE')
plt.title('Happiness: Within Cluster SSE vs Number of Clusters')
plt.legend(loc='best')
plt.show()
##print("Labels" + clf.labels_)
##print("Cluster Centers" + clf.cluster_centers_)

##train_prediction = clf.predict(traindata)
###trainaccuracy = accuracy_score(train_prediction, Y_train)*100
##print("The training accuracy for this is " +str(trainaccuracy))
##
###precision outcomes
##from sklearn.metrics import precision_score
##from sklearn.metrics import log_loss
##precision = precision_score(Y_test, Y_prediction, average = "weighted")*100
##loss = log_loss(Y_test, Y_prediction)*100
##print("Precision: " + str(precision))
##print("Loss: " + str(loss))
##
###Visualizations for model accuracy
###Learning Curve Estimator, Cross Validation

t1= time.clock()
time = str(t1-t0)
print("Computation Time" + time)

skplt.estimators.plot_learning_curve(clf, X, Y, title = "Happiness Learning Curve: K-Clusters")
plt.show()

##
###Visualization for kmeans
range_n_clusters = range(2,5)
y=Y
silhouetteScore = []
# for n_clusters in range_n_clusters:
#     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#     cluster_labels = clusterer.fit_predict(X)
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)
#     silhouetteScore.append(silhouette_avg)

# fig = plt.figure()
# plt.plot(range_n_clusters, silhouetteScore, 'r', label="SilhouetteScore Average")
# plt.xlabel('K Clusters')
# plt.ylabel('Silhouette Score')
# plt.title(': Average Silhouette Score vs Number of Clusters')
# plt.legend(loc='best')
# plt.show()



for n_clusters in range_n_clusters:

   
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()