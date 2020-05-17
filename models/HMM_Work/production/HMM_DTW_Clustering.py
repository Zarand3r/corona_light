#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import datetime
from dtaidistance import dtw
from dtaidistance import clustering
import json
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list
from matplotlib import pyplot as plt

import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir


def znormalize(ls):
#normalizes a list, if std=0 return the list
    std = np.std(ls)
    if std == 0.0:
        return np.array(ls)
    else:
        val = (ls - np.mean(ls))/np.std(ls)
        return (ls - np.mean(ls))/np.std(ls)

def znormalize_nozeros(ls):
#normalizes a list, if std=0 just pass
    std = np.std(ls)
    if std == 0.0:
        pass
    else:
        return (ls - np.mean(ls))/np.std(ls)

def noinf(arr):
    #Removes inf from list of lists
    newarr = []
    for x in arr:
        temp = x
        temp[temp == np.inf] = 9999
        newarr.append(x)
    return newarr

def nonzerofips(arr):
    #Takes in dataset, returns indices of data that do not have a list with all 0's
    ind = []
    for i in range(len(arr)):
        if np.std(arr[i]) != 0:
            ind.append(i)
    return ind

def makeZ(Data):
    #Creates DTW linkage matrix using DTAIdistance and scipy
    distance = dtw.distance_matrix_fast(Data,compact=True)
    Z = linkage(distance, method='complete')
    return Z

def fillnonzero(OrigData, clusters):
    #Takes a clustering from a dataset with nonzero entries.
    #Adds to that clustering another cluster for all 0's
    n = 0
    newclusters = []
    for i in range(len(OrigData)):
        if np.std(OrigData[i]) == 0:
            newclusters.append(0)
        else:
            newclusters.append(clusters[n])
            n += 1
    return newclusters


with open('NYT_daily_Warp_Death.txt') as f:
    NYT_daily_Warp_Death = json.load(f)
with open('NYT_daily_Death_Filled.txt') as g:
    NYT_daily_Death_Filled = json.load(g)
with open('JHU_daily_death.txt') as h:
    JHU_daily_death = json.load(h)
    
NYT_F = pd.read_csv(f"{homedir}/models/HMM_Work/production/NYT_daily_Filled.csv", index_col=0)
NYT_W = pd.read_csv(f"{homedir}/models/HMM_Work/production/NYT_daily_Warp.csv", index_col=0)
JHU = pd.read_csv(f"{homedir}/models/HMM_Work/production/JHU_daily.csv", index_col=0)

#Original dataset, making into list of np arrays
NYT_daily_Warp_Death = [np.array(x) for x in NYT_daily_Warp_Death]
NYT_daily_Death_Filled = [np.array(x) for x in NYT_daily_Death_Filled]
JHU_daily_death = [np.array(x) for x in JHU_daily_death]


#Z normalization of our dataset
Series_NYT_W = [znormalize(x) for x in NYT_daily_Warp_Death]
Series_NYT_F = [znormalize(x) for x in NYT_daily_Death_Filled]
Series_JHU = [znormalize(x) for x in JHU_daily_death]


#Removal of Strictly 0 lists from our dataset, these will belong in cluster 0
Series_NYT_W_nozeros = [znormalize_nozeros(x) for x in NYT_daily_Warp_Death]
Series_NYT_W_nozeros =  [x for x in Series_NYT_W_nozeros if x is not None]

Series_NYT_F_nozeros = [znormalize_nozeros(x) for x in NYT_daily_Death_Filled]
Series_NYT_F_nozeros =  [x for x in Series_NYT_F_nozeros if x is not None]

Series_JHU_nozeros = [znormalize_nozeros(x) for x in JHU_daily_death]
Series_JHU_nozeros =  [x for x in Series_JHU_nozeros if x is not None]


#We generate the many clusters needed for analysis
#Suffix "O": uses original unedited data
#"Z": uses z-normalized data, "N": uses z-normalized data, with all 0's entries in individual cluster
#"T": represents Tight, means a lower nubmer of clusters used
#"L": represents Loose, a higher number of clusters used
JHU_Cluster_Size = [2,2,6,2,6]

Z_JHU_O = makeZ(JHU_daily_death)
Z_JHU_Z = makeZ(Series_JHU)
Z_JHU_N = makeZ(Series_JHU_nozeros)

JHU_O = fcluster(Z_JHU_O, JHU_Cluster_Size[0], criterion ='maxclust')
JHU_Z_T = fcluster(Z_JHU_Z, JHU_Cluster_Size[1], criterion ='maxclust')
JHU_Z_L = fcluster(Z_JHU_Z, JHU_Cluster_Size[2], criterion ='maxclust')
JHU_N_T = fillnonzero(Series_JHU,fcluster(Z_JHU_N, JHU_Cluster_Size[3], criterion ='maxclust'))
JHU_N_L = fillnonzero(Series_JHU,fcluster(Z_JHU_N, JHU_Cluster_Size[4], criterion ='maxclust'))

ClustersJHU = pd.DataFrame(data=JHU.FIPS.unique(),columns=['FIPS'])
ClustersJHU['JHU_Orig'] = JHU_O
ClustersJHU['JHU_Z_T'] = JHU_Z_T
ClustersJHU['JHU_Z_L'] = JHU_Z_L
ClustersJHU['JHU_N_T'] = JHU_N_T
ClustersJHU['JHU_N_L'] = JHU_N_L

NYT_F_Cluster_Size = [2,2,5,2,5,9]

Z_NYT_F_O = makeZ(NYT_daily_Death_Filled)
Z_NYT_F_Z = makeZ(Series_NYT_F)
Z_NYT_F_N = makeZ(Series_NYT_F_nozeros)

NYT_F_O = fcluster(Z_NYT_F_O, NYT_F_Cluster_Size[0], criterion ='maxclust')
NYT_F_Z_T = fcluster(Z_NYT_F_Z, NYT_F_Cluster_Size[1], criterion ='maxclust')
NYT_F_Z_L = fcluster(Z_NYT_F_Z, NYT_F_Cluster_Size[2], criterion ='maxclust')
NYT_F_N_T = fillnonzero(Series_NYT_F,fcluster(Z_NYT_F_N, NYT_F_Cluster_Size[3], criterion ='maxclust'))
NYT_F_N_L = fillnonzero(Series_NYT_F,fcluster(Z_NYT_F_N, NYT_F_Cluster_Size[4], criterion ='maxclust'))
NYT_F_N_L_L = fillnonzero(Series_NYT_F,fcluster(Z_NYT_F_N, NYT_F_Cluster_Size[5], criterion ='maxclust'))

ClustersNYT_F = pd.DataFrame(data=NYT_F.fips.unique(),columns=['FIPS'])
ClustersNYT_F['NYT_F_Orig'] = NYT_F_O
ClustersNYT_F['NYT_F_Z_T'] = NYT_F_Z_T
ClustersNYT_F['NYT_F_Z_L'] = NYT_F_Z_L
ClustersNYT_F['NYT_F_N_T'] = NYT_F_N_T
ClustersNYT_F['NYT_F_N_L'] = NYT_F_N_L
ClustersNYT_F['NYT_F_N_L_L'] = NYT_F_N_L_L


NYT_W_Cluster_Size = [2,2,7,2,7]

Z_NYT_W_O = makeZ(NYT_daily_Warp_Death)
Z_NYT_W_Z = makeZ(Series_NYT_W)
Z_NYT_W_N = makeZ(Series_NYT_W_nozeros)

NYT_W_O = fcluster(Z_NYT_W_O, NYT_W_Cluster_Size[0], criterion ='maxclust')
NYT_W_Z_T = fcluster(Z_NYT_W_Z, NYT_W_Cluster_Size[1], criterion ='maxclust')
NYT_W_Z_L = fcluster(Z_NYT_W_Z, NYT_W_Cluster_Size[2], criterion ='maxclust')
NYT_W_N_T = fillnonzero(Series_NYT_W,fcluster(Z_NYT_W_N, NYT_W_Cluster_Size[3], criterion ='maxclust'))
NYT_W_N_L = fillnonzero(Series_NYT_W,fcluster(Z_NYT_W_N, NYT_W_Cluster_Size[4], criterion ='maxclust'))

ClustersNYT_W = pd.DataFrame(data=NYT_W.fips.unique(),columns=['FIPS'])
ClustersNYT_W['NYT_W_Orig'] = NYT_W_O
ClustersNYT_W['NYT_W_Z_T'] = NYT_W_Z_T
ClustersNYT_W['NYT_W_Z_L'] = NYT_W_Z_L
ClustersNYT_W['NYT_W_N_T'] = NYT_W_N_T
ClustersNYT_W['NYT_W_N_L'] = NYT_W_N_L


AllClusters = ClustersJHU.join(ClustersNYT_F.set_index('FIPS'), on='FIPS', how='outer').join(ClustersNYT_W.set_index('FIPS'), on='FIPS', how='outer').sort_values('FIPS')

AllClusters.to_csv('DTW_Clustering.csv')

