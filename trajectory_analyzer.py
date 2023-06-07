from trajectory import Trajectory
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D


class TrajectoryAnalizer:
    def __init__(self, trj:Trajectory):
        
        # kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(trj.points)
        # print(kmeans.labels_)

        data = trj.points

        model = DBSCAN(eps=2.5, min_samples=2)
        model.fit_predict(data)
        pred = model.fit_predict(data)

        trj.clusters = model.labels_
        print("number of cluster found: {}".format(len(set(model.labels_))))
        print('cluster for each point: ', model.labels_)
