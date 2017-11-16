from __future__ import print_function

import sys

import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
spark = SparkSession\
        .builder\
        .appName("PythonKMeans")\
        .getOrCreate()
path = "/home/immd-user/assignemt datasets/dataset-problemset3-ex1-2.npy"
def distance(x,y):
    return np.sqrt(np.sum((x[0]-y[0])**2+(x[1]-y[1])**2))

a = np.load(path)
N = np.shape(a)[0]
sc = spark.sparkContext
points = sc.parallelize(a)
def g(x):
    print(x)
clusters = []
clustersdictsumandcount = {}#which stores x cordinate of centroid as key and sum and count of the respective lcuste
clustersdict = {}#which stroes x corordinate of centroid as key and points in cluster as keys

indexpoints = points.zipWithIndex()
indexpoints.foreach(g)
while indexpoints.count()>1:
    indexpoints = indexpoints.map(lambda x : (np.around(x[0],2),x[1]))
    allpoints = indexpoints.cartesian(indexpoints).map(lambda x :(distance(x[0][0],x[1][0]),(x[0][0].tolist(),x[1][0].tolist()),(x[0][1],x[1][1])))
    distancesrdd = allpoints.filter(lambda x : x[2][0]!=x[2][1])
    firstbest = distancesrdd.sortBy(lambda x : x[0] ).first()
    if tuple(firstbest[1][0]) in clustersdict.keys():
        sum = np.array(clustersdictsumandcount[tuple(firstbest[1][0])][0])+np.array(firstbest[1][1])
        count = clustersdictsumandcount[tuple(firstbest[1][0])][1]+1
        centroid = np.around(np.true_divide(sum,count),2)
        centroid = centroid.tolist()
        newcluster = [clustersdict[tuple(firstbest[1][0])],firstbest[1][1]]

        clusters.append(newcluster)
        clustersdict[tuple(centroid)] = newcluster
        clustersdictsumandcount[tuple(centroid)] = (sum,count)
        clusters.remove(clustersdict[tuple(firstbest[1][0])])
        clustersdict.pop(tuple(firstbest[1][0]))
        clustersdictsumandcount.pop(tuple(firstbest[1][0]))
        
        N = N+1
    elif tuple(firstbest[1][1]) in clustersdict.keys():
        sum = np.array(clustersdictsumandcount[tuple(firstbest[1][1])][0])+np.array(firstbest[1][0])
        count = clustersdictsumandcount[tuple(firstbest[1][1])][1]+1
        centroid = np.around(np.true_divide(sum,count),2)
        centroid = centroid.tolist()
        newcluster = [clustersdict[tuple(firstbest[1][1])],firstbest[1][0]]
        clusters.append(newcluster)
        clustersdict[tuple(centroid)] = newcluster
        clustersdictsumandcount[tuple(centroid)] = (sum,count)
        clusters.remove(clustersdict[tuple(firstbest[1][1])])
        clustersdict.pop(tuple(firstbest[1][1]))
        clustersdictsumandcount.pop(tuple(firstbest[1][1]))
        N=N+1
    else:     
        centroid = (np.array(firstbest[1][0])+np.array(firstbest[1][1]))/2
        centroid = np.around(centroid,2).tolist()
        clustersdict[tuple(centroid)] = np.true_divide(firstbest[1][0]+firstbest[1][1],2)   
        
        sum  = np.array(firstbest[1][0])+np.array(firstbest[1][1])
        count = 2
        newcluster = [firstbest[1][0],firstbest[1][1]]
        clusters.append(newcluster)
        clustersdict[tuple(centroid)] = newcluster
        clustersdictsumandcount[tuple(centroid)] = (sum,count)
        N = N+1
    print("------------------------clusters------------------")
    print(clusters)
    outcluster = indexpoints.filter(lambda i: i[1]!=firstbest[2][0] and i[1]!=firstbest[2][1] )
    outcluster = outcluster.union(sc.parallelize([tuple([np.array(centroid),N])]))
    indexpoints = outcluster

