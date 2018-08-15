import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import random
from tqdm import tqdm
import numpy as np
import statsmodels.api as sm

from sklearn.manifold import MDS
from mpl_toolkits import mplot3d
from scipy.spatial import distance
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans, AffinityPropagation, DBSCAN, FeatureAgglomeration
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from IPython.display import clear_output, Image, display
from sklearn.datasets.samples_generator import make_blobs
import itertools
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import os
plt.ion()
plt.show()

printFunctionNames = True

if printFunctionNames:
    print('elbowAnalysis')
def elbowAnalysis(X, numberOfClusters):
    distortions = []

    for k in tqdm(numberOfClusters):
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    plt.plot(numberOfClusters, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    
    
if printFunctionNames:
    print('silhouetteAnalyis')
def silhouetteAnalyis (X, numberOfClusters):
    silhouette_score_values=[]
    for i in tqdm(numberOfClusters):
        classifier=KMeans(i,init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True)
        classifier.fit(X)
        labels= classifier.predict(X)
        silhouette_score_values.append(metrics.silhouette_score(X,labels ,metric='euclidean', sample_size=None, random_state=None))

    plt.plot(numberOfClusters, silhouette_score_values)
    plt.title("Silhouette score values vs Numbers of Clusters ")
    plt.show()

    Optimal_NumberOf_Components=numberOfClusters[silhouette_score_values.index(max(silhouette_score_values))]
    print( "Optimal number of components is:", Optimal_NumberOf_Components)
    
if printFunctionNames:
    print('bicAicAnalysis')
def bicAicAnalysis(X, numberOfClusters):
    bic = []
    aic = []

    for n in tqdm(numberOfClusters):
        model = GaussianMixture(n, covariance_type ='full', random_state = 0).fit(X)
        bic.append(model.bic(X))
        aic.append(model.aic(X))

    plt.plot(numberOfClusters, bic, label = 'BIC')
    plt.plot(numberOfClusters, aic, label = 'AIC')
    plt.legend()
    plt.title('BIC/AIC')
    plt.xlabel('n_components')
    
if printFunctionNames:
    print('matrixToOnes')    
def matrixToOnes(a, threshold =0):
    b = np.zeros(a.shape)
    x, y = np.where(a>threshold)
    b[x, y] = 1
    return b

if printFunctionNames:
    print('overlap')
def overlap(a, threshold = 0, printOverlap = False):
    """
    This method will convert values in a > threshold to 1 and the others to 0
    Perfect overap is the number of expressed genes
    """
    a =  matrixToOnes(a, threshold)
    o = np.sum(a, axis = 0) /a.shape[0]
    if printOverlap:
        print('Overlap vector %s \n, %s\n' % (o, Counter(o)))
    perfectOverlap = Counter(o)[1]/a.shape[0] 
    overallOverlap = np.sum(o)/a.shape[0]
    return perfectOverlap, overallOverlap

if printFunctionNames:
    print('customDistance')
def customDistance(a, b, name ='L2'):
    if name == 'L2':
        return distance.euclidean(a, b)

if printFunctionNames:
    print('intraClusterWeights')
def intraClusterWeights(a, b, dist='L2'):
    result = 0
    for i, j in itertools.product(range(len(a)), range(len(b))):
        result += customDistance(a[i], b[j], dist)

    if (a == b).all():
#         print('Calculating Intracluster Distance')
        result /= 2
    return result

if printFunctionNames:
    print('daviesBouldin')
def daviesBouldin(X, labels):
    labels = np.array(labels)
    n_cluster = len(np.bincount(labels))
    cluster_k = [X[labels == k] for k in range(n_cluster)]
    centroids = [np.mean(k, axis = 0) for k in cluster_k]
    variances = [np.mean([distance.euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
    db = []

    for i in range(n_cluster):
        for j in range(n_cluster):
            if j != i:
                db.append((variances[i] + variances[j]) / distance.euclidean(centroids[i], centroids[j]))
    return(np.max(db) / n_cluster)

if printFunctionNames:
    print('clusterDetails')
def clusterDetails(a, threshold =0 , printOverlap = False):
    """
    For gien cluster, this method returns:
    - array of std
    - array of mean values
    - top k expressed genes
    - other internal cluster measures
    
    """

    stDev = np.std(a, axis = 0)
    avg = np.mean(a, axis = 0)
#     avg.argsort()[-k:][::-1]
    intraclusterDistance = intraClusterWeights(a, a)/ len(a)
    expressedGenes = np.unique(np.where(a>threshold)[1])
    a =  matrixToOnes(a, threshold)
    
    o = np.sum(a, axis = 0)/a.shape[0]
    if printOverlap:
        print('Overlap vector %s ' % o)
    perfectOverlapGenes = np.where(a.all(axis=0))[0]

    return { 'genesStd' : stDev, 
            'genesAvg' : avg,
            'intraclusterDistance' :intraclusterDistance,
            'expressedGenes': expressedGenes,
           'perfectOverlapGenes' : perfectOverlapGenes,
           'numCells': len(a)}

if printFunctionNames:
    print('evaluateClusters')
def evaluateClusters(a, clusters, teta = 0):
    intraClusterData = {}
    for clusterId in tqdm(np.unique(clusters)):
        rowIds = np.where(clusters == clusterId)[0]
        cluster = a[rowIds]
#         clusterData[clusterId] = overlap(cluster, teta=teta)
        intraClusterData[clusterId] = clusterDetails(cluster)
    interClusterData = {}
    silhouetteScore = metrics.silhouette_score(a, clusters, metric='euclidean')
    interClusterData['silhouetteScore']= silhouetteScore
    interClusterData['daviesBouldin']= daviesBouldin(a, clusters)
    return intraClusterData, interClusterData

if printFunctionNames:
    print('visualizeEvalData')
def visualizeEvalData(clusterEvalData, name):
    print(name + ' : Inter cluster measures : ', clusterEvalData[1])
    visualizeIntraCluster(clusterEvalData[0], name)
    

if printFunctionNames:
    print('visualizeIntraCluster')
def visualizeIntraCluster(clusterEvalData, name):
    numClusters = len(clusterEvalData.keys())
    hvalue = [x['numCells'] for x in clusterEvalData.values()]

    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.hist(hvalue, bins=numClusters)
    plt.title('Histogram');
    plt.subplot(122)
    plt.bar(np.arange(len(hvalue)), sorted(hvalue))
    plt.title('sorted values')
    plt.suptitle(name + ' Number of cells per cluster')


    hvalue = np.array([ -1 if x['numCells'] == 1 else np.mean(x['genesStd'])*1 for x in clusterEvalData.values()])
    hvalue[hvalue==-1] = -1*np.min(hvalue[(hvalue>0)])


    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.hist(hvalue, bins=numClusters)
    plt.title('Histogram');
    plt.subplot(122)
    plt.bar(np.arange(len(hvalue)), sorted(hvalue))
    plt.title('values')
    plt.xlabel('cluster Number')
    plt.suptitle(name +  ' mean(std(geneValues)) per Cluster \n[-1 if only 1 value]')


    hvalue = [len(x['perfectOverlapGenes']) for x in clusterEvalData.values() if x['numCells'] != 1]

    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.hist(hvalue, bins=numClusters)
    plt.title('Histogram');
    plt.subplot(122)
    if np.sum(hvalue) != 0:
        plt.bar(np.arange(len(hvalue)), sorted(hvalue), log = True, label = 'Clusters with more than 1 cell', alpha = 0.4)
    plt.legend()
    plt.title('values')
    plt.xlabel('cluster Number')
    plt.ylabel('Logscale')
    plt.suptitle(name + ' Number of common genes expressed in ALL cells within cluster (perfect overlap) for clusters with more than 1 cell\n Threshold dependent!!')

    hvalue = np.array([ -1 if x['numCells'] == 1 else x['intraclusterDistance'] for x in clusterEvalData.values()])
    hvalue[hvalue==-1] = -1*np.min(hvalue[(hvalue>0)])

    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.hist(hvalue, bins=numClusters)
    plt.title('Histogram');
    plt.subplot(122)
    plt.bar(np.arange(len(hvalue)), sorted(hvalue), log = True)
    plt.title('values')
    plt.xlabel('cluster Number')
    plt.ylabel('Logscale of intra cluster distance')
    plt.suptitle(name + ' Intra cluster distance L2')
    
if printFunctionNames:
    print('saveCluster')    
def saveCluster(cluster, name):
    fname = 'data/clusters/cluster_%s.npy' % name
    np.save(fname,cluster)
    
if printFunctionNames:
    print('generateClusters')    
def generateClusters(X, clusterType, nbClusters, name):
    for n in tqdm(nbClusters):
        if clusterType == 'KMeans':
            kmeans = KMeans(n_clusters=n)
            kmeans.fit(X)
            clusters = kmeans.predict(X)
            saveCluster(clusters, str(n) + '_kmeans_' + name)
            
        if clusterType == 'GaussianMixture':
            kmeans = GaussianMixture(n, covariance_type ='full', random_state = 0)
            kmeans.fit(X)
            clusters = kmeans.predict(X)
            saveCluster(clusters, str(n) + '_kmeans_' + name)

if printFunctionNames:
    print('loadAndEvaluateCluster')
def loadAndEvaluateCluster(name, df, redoEvaluation=False, isFileName = False):
    fname = 'data/clusters/cluster_%s.npy' % name
    if isFileName:
        fname = name
    if os.path.isfile(fname):
        c = np.load(fname)
    else:
        return None
    
    fname = 'data/clusterEval/eval_%s.pkl' % name
    if os.path.isfile(fname) and not redoEvaluation:
        e = pickle.load(open(fname,'rb'))
    else:
        e = evaluateClusters(df, c)
        pickle.dump(e, open(fname, 'wb'))
    return c, e