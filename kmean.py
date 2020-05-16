import subprocess
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
import PyGnuplot as gp
import random
import time
from copy import deepcopy


# Euclidean distanceance caculator
def distance(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


def dist(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def random_centroids(n, k):
    # X coordinates of random centroids
    C_x = np.random.randint(0, n, size=k)
    return C_x


def plotData(req):
    # g = Gnuplot.Gnuplot(persist =1)
    if req == 1:
        gp.c('set terminal qt 2 title "Logistic function"')
    else:
        gp.c('set terminal qt 2 title "Neural Network"')
    gp.c("set palette model RGB defined ( 0 'red', 1 'green', 2 'blue', 3 'black')")
    gp.c("unset key")
    gp.c('set xrange [-2:2]')
    gp.c('set yrange [-2:2]')
    gp.c('plot "p2.dat" using 2:3:1 with points palette')


def printFile(file, data, clusters):
    f = open(file, "w")
    for i in range(len(data)):
        s = str(clusters[i]) + " " + str(data[i][0]) + " " + str(data[i][1]) + "\n"
        f.write(s)
    f.close()

def readTrainingData(file):
    X = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            point = [float(j) for j in line.split()]
            X.append(point)
    data = np.array(X)
    plotData(file)
    time.sleep(0.25)
    data = data[:, 1:]
    f.close()
    # plt.scatter(data[:,0],data[:,1], c = 'black', s=7)
    # plt.show()
    return data


def printGroup(data, clusters):
    f = open("new.plt", "w")
    for i in range(len(data)):
        s = str(clusters[i]) + " " + str(data[i][0]) + " " + str(data[i][1]) + "\n"
        f.write(s)
    f.close()
    gp.c('set terminal qt size 500,400 title "Kmeans"')
    gp.c("set palette model RGB defined ( 0 'red', 1 'green', 2 'blue', 3 'black')")
    gp.c("unset key")
    gp.c('plot "new.plt" using 2:3:1 with points palette')


def inter_class(nb, centroids, g):
    I_b = 0
    for i in range(len(nb)):
        d = dist(centroids[i], g)
        I_b += nb[i] * d * d
    return I_b


def gen_data(n):
    X = []
    group1 = np.random.uniform(low=0, high=1, size=(n, 2))
    for x, y in group1: X.append([0, 1 + x, 1 + y])
    for x, y in group1: X.append([0, 1 + x, -2 + y])
    for x, y in group1: X.append([0, -2 + x, 1 + y])
    for x, y in group1: X.append([0, -2 + x, -2 + y])
    data = np.array(X)
    clusters = data[:, 0]
    data = data[:, 1:]
    printGroup(data, clusters)
    time.sleep(0.25)
    return data, clusters


def kmean(k, data):
    n = len(data)
    iteration = 5
    res_clusters = []
    nb_cluster = [0 for i in range(k)]
    clusters = np.zeros(len(data))
    I_b = 0
    I_b_max = 0
    while (iteration > 0):
        # inittial random center
        centroids = data[np.random.choice(data.shape[0], k, replace=False)]
        while True:
            # assign observation
            for i in range(len(data)):
                dist_centroids = [dist(data[i], x) for x in centroids]
                # dist_centroids = distance(data[i], centroids)
                cluster = np.argmin(dist_centroids)
                clusters[i] = cluster
            # recalculate new centroids
            old_ib = I_b
            for i in range(k):
                groups = [data[j] for j in range(len(data)) if clusters[j] == i]
                nb_cluster[i] = len(groups)
                centroids[i] = np.mean(groups, axis=0)
            printGroup(data, clusters)
            time.sleep(0.25)
            # calculate the mean vector G of the centroids
            g = np.mean(centroids, axis=0)
            I_b = inter_class(nb_cluster, centroids, g)
            print("g ", g, " i_b ", inter_class(nb_cluster, centroids, g))
            # error = distance(clusters, old_c, None)
            if (I_b <= old_ib): break
        if (I_b > I_b_max):
            I_b_max = I_b
            res_clusters = deepcopy(clusters)
        iteration -= 1

    print("k = ", k, " ", res_clusters)
    printGroup(data, res_clusters)
    time.sleep(0.25)
