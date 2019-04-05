import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



filename = sys.argv[1]
filename2 = sys.argv[2]
char_data = pd.read_csv(filename, sep=',', header=None, low_memory=False)

#char_data = pd.read_csv("happy.csv", sep = ',', header = None, low_memory = False)
char_x = char_data.values[1:, 2:]


NoNoise_data = pd.read_csv(filename, sep = ',', header = None, low_memory = False)
#NoNoise_data = pd.read_csv("stars.csv", sep = ',', header = None, low_memory = False)
NoNoise_x = NoNoise_data.values[1:, [0,1]]


wine_data = pd.read_csv(filename2, sep = ',', header = None, low_memory = False)
#wine_data = pd.read_csv("stars.csv", sep = ',', header = None, low_memory = False)
wine_x = wine_data.values[1:, 1:]

#pca transformation
char_x = PCA(n_components=2).fit_transform(char_x)
NoNoise_x = PCA(n_components=2).fit_transform(NoNoise_x)
wine_x = PCA(n_components=2).fit_transform(wine_x)


K = range(1, 40)
KM_c = [KMeans(init = 'k-means++', n_clusters=k).fit(char_x) for k in K]
KM_w = [KMeans(n_clusters=k).fit(wine_x) for k in K]
KM_n = [KMeans(n_clusters=k).fit(NoNoise_x) for k in K]
print("Trained kmean models")
centroids_c = [km.cluster_centers_ for km in KM_c]
centroids_w = [km.cluster_centers_ for km in KM_w]
centroids_n = [km.cluster_centers_ for km in KM_n]
print("Found the centroids")
Dk_c = [cdist(char_x, center, 'euclidean') for center in centroids_c]
Dk_w = [cdist(wine_x, center, 'euclidean') for center in centroids_w]
Dk_n = [cdist(NoNoise_x, center, 'euclidean') for center in centroids_n]
print("Calculated euclidean distance")

cIdx_n = [np.argmin(D, axis=1) for D in Dk_n]
dist_n = [np.min(D, axis=1) for D in Dk_n]
avgWithinSS_n = [sum(d) / NoNoise_x.shape[0] for d in dist_n]
# Total with-in sum of square
wcss_n = [sum(d**2) for d in dist_n]
tss_n = sum(pdist(NoNoise_x)**2) / NoNoise_x.shape[0]
bss_n = tss_n - wcss_n

cIdx_c = [np.argmin(D, axis=1) for D in Dk_c]
dist_c = [np.min(D, axis=1) for D in Dk_c]
avgWithinSS_c = [sum(d) / char_x.shape[0] for d in dist_c]
# Total with-in sum of square
wcss_c = [sum(d**2) for d in dist_c]
tss_c = sum(pdist(char_x)**2) / char_x.shape[0]
bss_c = tss_c - wcss_c

cIdx_w = [np.argmin(D, axis=1) for D in Dk_w]
dist_w = [np.min(D, axis=1) for D in Dk_w]
avgWithinSS_w = [sum(d) / wine_x.shape[0] for d in dist_w]
# Total with-in sum of square
wcss_w = [sum(d**2) for d in dist_w]
tss_w = sum(pdist(wine_x)**2) / wine_x.shape[0]
bss_w = tss_w - wcss_w


print("Calculated sum of square errors")
kIdx_c = 9
kIdx_w = 1
kIdx_n = 1
plt.style.use('ggplot')
# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(K, avgWithinSS_c, '*-', label='Happiness Dataset Post DR')
# ax.plot(K[kIdx_c], avgWithinSS_c[kIdx_c], marker='o', markersize=12,
#         markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')


ax.plot(K, avgWithinSS_w, '*-', label='Kepler Star Dataset Post DR')
# ax.plot(K[kIdx_w], avgWithinSS_w[kIdx_w], marker='o', markersize=12,
#         markeredgewidth=2, markeredgecolor='b', markerfacecolor='None')

#ax.plot(K, avgWithinSS_n, '*-', label='Kepler Star Dataset Without Noise')
# ax.plot(K[kIdx_n], avgWithinSS_n[kIdx_n], marker='o', markersize=12,
#         markeredgewidth=2, markeredgecolor='b', markerfacecolor='None')


plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.legend(loc='best')
plt.title('Elbow for KMeans clustering Post DR')
# plt.title('Elbow for KMeans clustering')
# fig.savefig('graphs/kmeans/elbow1.png')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, bss_c / tss_c * 100, '*-', label='Happiness Rating')
ax.plot(K, bss_w / tss_w * 100, '*-', label='Likelihood of Host Star')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.legend(loc='best')
plt.title('Elbow for KMeans clustering')
# fig.savefig('graphs/kmeans/elbow2.png')
plt.show()











# #Decision Trees
# from __future__ import print_function
#
# #info from http://scikit-learn.org/stable/modules/
# #generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
#
# #import sklearn statements
# import sklearn as sklearn
# from sklearn.metrics import accuracy_score
# from sklearn import preprocessing
# from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import learning_curve
# from sklearn.model_selection import train_test_split
#
# #for graph from http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
# from sklearn.metrics import silhouette_samples, silhouette_score
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# print(__doc__)
#
# #k-means specific imports
# from sklearn.cluster import KMeans
# from sklearn.metrics import accuracy_score
#
# #other imports
# import scikitplot as skplt
# import matplotlib.pyplot as plt
# import csv
# import numpy as np
# import pandas as pd
# import time
# from sklearn.model_selection import validation_curve
# from datetime import date
#
# #Read data in using pandas
#
# trainDataSet = pd.read_csv("stars.csv", sep = ',', header = None, low_memory = False)
#
# #encode text data to integers using getDummies
# traindata = pd.get_dummies(trainDataSet)
# traindata= traindata[:1000]
# # Create decision Tree using major_category, month, year, to predict violent or not
# # train split uses default gini node, split using train_test_split
#
# X = trainDataSet.values[1:, 1:]
# Y = trainDataSet.values[1:,0]
# # X = trainDataSet.values[1:, [5,12]]
# # Y = trainDataSet.values[1:,20]
#
# #start timer
# t0= time.clock()
#
# #cv = train_test_split(X, Y, test_size=.30, random_state= 20)
# ##X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state= 20)
# ##
#
# # k = 2
# # clf = sklearn.cluster.KMeans(n_clusters= k, init='k-means++', n_init = 5, max_iter = 500)
#
# # clf = clf.fit(traindata)
# ##print("Labels" + clf.labels_)
# ##print("Cluster Centers" + clf.cluster_centers_)
#
# ##train_prediction = clf.predict(traindata)
# ###trainaccuracy = accuracy_score(train_prediction, Y_train)*100
# ##print("The training accuracy for this is " +str(trainaccuracy))
# ##
# ###precision outcomes
# ##from sklearn.metrics import precision_score
# ##from sklearn.metrics import log_loss
# ##precision = precision_score(Y_test, Y_prediction, average = "weighted")*100
# ##loss = log_loss(Y_test, Y_prediction)*100
# ##print("Precision: " + str(precision))
# ##print("Loss: " + str(loss))
# ##
# ###Visualizations for model accuracy
# ###Learning Curve Estimator, Cross Validation
#
#
#
# range_n_clusters = range(2,30)
# y=Y
# silhouetteScore = []
# for n_clusters in range_n_clusters:
#     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#     cluster_labels = clusterer.fit_predict(X)
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)
#     silhouetteScore.append(silhouette_avg * 100)
#
# fig = plt.figure()
# plt.plot(range_n_clusters, silhouetteScore, 'r', label="SilhouetteScore Average")
# plt.xlabel('K Clusters')
# plt.ylabel('Within Cluster SSE')
# plt.title('Stars: Within Cluster SSE vs Number of Clusters')
# plt.legend(loc='best')
# plt.show()
#
#
# t1= time.clock()
# time = str(t1-t0)
# print("Computation Time" + time)
#
# # skplt.estimators.plot_learning_curve(clf, X, Y, title = "Cancer Learning Curve: K-Clusters")
# # plt.show()
#
# ##
# ###Visualization for kmeans
# range_n_clusters = range(2,5)
# y=Y
# silhouetteScore = []
# # for n_clusters in range_n_clusters:
# #     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
# #     cluster_labels = clusterer.fit_predict(X)
# #     silhouette_avg = silhouette_score(X, cluster_labels)
# #     print("For n_clusters =", n_clusters,
# #           "The average silhouette_score is :", silhouette_avg)
# #     silhouetteScore.append(silhouette_avg)
#
# # fig = plt.figure()
# # plt.plot(range_n_clusters, silhouetteScore, 'r', label="SilhouetteScore Average")
# # plt.xlabel('K Clusters')
# # plt.ylabel('Silhouette Score')
# # plt.title('Cancer: Average Silhouette Score vs Number of Clusters')
# # plt.legend(loc='best')
# # plt.show()
#
#
#
# for n_clusters in range_n_clusters:
#     # Create a subplot with 1 row and 2 columns
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     # fig.set_size_inches(18, 7)
#
#     # # The 1st subplot is the silhouette plot
#     # # The silhouette coefficient can range from -1, 1 but in this example all
#     # # lie within [-0.1, 1]
#     ax1.set_xlim([-0.1, 1])
#     # # The (n_clusters+1)*10 is for inserting blank space between silhouette
#     # # plots of individual clusters, to demarcate them clearly.
#     ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
#
#     # Initialize the clusterer with n_clusters value and a random generator
#     # seed of 10 for reproducibility.
#     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#     cluster_labels = clusterer.fit_predict(X)
#     plt.scatter(X[:, 0], X[:, 1], c=cluster_labels)
#     plt.title("Gender with Noise K-Means Clustering")
#     plt.show()
#     # The silhouette_score gives the average value for all the samples.
#     # This gives a perspective into the density and separation of the formed
#     # clusters
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)
#
#     # Compute the silhouette scores for each sample
#     sample_silhouette_values = silhouette_samples(X, cluster_labels)
#
#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belonging to
#         # cluster i, and sort them
#         ith_cluster_silhouette_values = \
#             sample_silhouette_values[cluster_labels == i]
#
#         ith_cluster_silhouette_values.sort()
#
#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i
#
#         color = cm.spectral(float(i) / n_clusters)
#         ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                           0, ith_cluster_silhouette_values,
#                           facecolor=color, edgecolor=color, alpha=0.7)
#
#         # Label the silhouette plots with their cluster numbers at the middle
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
#
#         # Compute the new y_lower for next plot
#
#     ax1.set_title("The silhouette plot for the various clusters.")
#     ax1.set_xlabel("The silhouette coefficient values")
#     ax1.set_ylabel("Cluster label")
#
#     # The vertical line for average silhouette score of all the values
#     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
#
#     ax1.set_yticks([])  # Clear the yaxis labels / ticks
#     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
#
#     # 2nd Plot showing the actual clusters formed
#     colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
#     ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
#                 c=colors, edgecolor='k')
#
#     # Labeling the clusters
#     centers = clusterer.cluster_centers_
#     # Draw white circles at cluster centers
#     ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
#                 c="white", alpha=1, s=200, edgecolor='k')
#
#     for i, c in enumerate(centers):
#         ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
#                     s=50, edgecolor='k')
#
#     ax2.set_title("The visualization of the clustered data.")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")
#
#     plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
#                   "with n_clusters = %d" % n_clusters),
#                  fontsize=14, fontweight='bold')
#
#     plt.show()
