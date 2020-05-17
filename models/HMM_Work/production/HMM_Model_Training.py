#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import datetime
from HMM import unsupervised_HMM
from HMM import supervised_HMM
from HMM_helper import sample_sentence
import json
from hmmlearn import hmm
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir


def makeHMMUnSupData(Input, colname, fipsname):
    #Takes input dataframe, and gives out HMM format of data, a list of lists 
    #of the colname value, each list in the set represents one fips code.
    Output = []
    for fips in Input[fipsname].unique():
        temp = list(Input[Input[fipsname] == fips][colname])
        Output.append(temp)
    return Output

def makeHMMmap(Output):
    #Takes in output of makeHMMUnSupData and transforms data into list from 0 to D-1, where D is the number of unique
    #values of the output
    #Unqiue values in the input
    UniqueVals = np.array(list(set(x for l in Output for x in l)))
    UniqueVals = np.sort(UniqueVals)
    HMMOutput = []
    templs = []
    Map = {}
    RMap = {}
    for x in range(len(UniqueVals)):
        Map[int(UniqueVals[x])] = x
        RMap[x] = int(UniqueVals[x])
    for ls in Output:
        for val in ls:
            templs.append(Map[val])
        HMMOutput.append(templs)
        templs = []
    return [Map,RMap,HMMOutput]

def makeHMMSupData(UnSupData):
    #Takes list of lists of time series data from makeHMMUnSupData and makes it into data with X and Y
    X = []
    Y = []
    tempX = []
    tempY = []
    for ls in UnSupData:
        lenls = len(ls)
        for n in range(lenls):            
            if n == 0:
                tempX.append(ls[n])
            elif n == lenls - 1:
                tempY.append(ls[n])
            else:
                tempX.append(ls[n])
                tempY.append(ls[n])
        if len(tempX) != 0 and len(tempY) != 0:
            X.append(tempX)
            Y.append(tempY)
        tempX = []
        tempY = []   
    return [X,Y]

def makeX(Data, DTW, cluster_col, cluster_num, fipsname, deathsname):
    #Takes in the dataset, cluster column and number, and gives out the deaths info in this cluster
    #In the form able to be processed by hmmlearn's HMM modules    
    fips = list(DTW[DTW[cluster_col] == cluster_num]['FIPS'])
    Rows = Data[Data[fipsname].isin(fips)]
    RawData = makeHMMUnSupData(Rows, deathsname, fipsname)
    #RawData = [a[0] for a in RawData]
    temp = []
    lengths = []
    for i in RawData:
        temp.extend(i)
        lengths.append(len(i))
    temp = np.array(temp).reshape(-1,1)
    return [temp, lengths]

def makeHMM(X):
    #Takes in data from makeX, and uses the Elbow method to determine the optimal number of 
    #states needed in the HMM, and returns the HMM with that optimal number of states
    scores = []
    Flag = True
    val = 999
    for i in range(1,31):
        tempmodel = hmm.GaussianHMM(n_components=i, covariance_type="full")
        #Tries to make the model fit, can fail if data not diverse enough
        try:
            if Flag:
                tempmodel.fit(X[0],X[1])
                scores.append(tempmodel.score(X[0],X[1]))
                if i > 10:
                    if scores[-1] < scores[-2]:
                        Flag = False
                print(i)
        except:
            val = i - 1
            Flag = False
    #If the data only accepts less than 4 states to work, we chose the max number of states to describe it
    if val < 5:
        return hmm.GaussianHMM(n_components = val, covariance_type="full").fit(X[0],X[1])
    else:
    #We do an elbow method otherwise
        n = 0
        #finding number of negative entries
        for j in scores:
            if j < 0:
                n += 1
        #gettin index of best point by elbow method (using first derivative)
        print(scores)
        ind = np.argmax(np.diff(scores)[(n + 1):]/scores[(n + 2):])
        return hmm.GaussianHMM(n_components = ind + n + 3, covariance_type="full").fit(X[0],X[1])


def makeHMMlist(Data, DTW, cluster_col):
    labels = np.sort(DTW[cluster_col].dropna().unique())
    HMM_list = [0] * len(labels)
    n = 0
    for i in labels:
        print(i)
        X = makeX(Data, DTW, cluster_col, i)
        HMM_list[n] = makeHMM(X)
        n += 1
    return [HMM_list, labels]



#Dataframes of deaths
NYT_F = pd.read_csv(f"{homedir}/models/HMM_Work/NYT_daily_Filled.csv", index_col=0)
NYT_W = pd.read_csv(f"{homedir}/models/HMM_Work/NYT_daily_Warp.csv", index_col=0)
NYT_F = NYT_F.rename(columns={'fips':'FIPS','deaths':'Deaths'})
NYT_W = NYT_W.rename(columns={'fips':'FIPS','deaths':'Deaths'})
JHU = pd.read_csv(f"{homedir}/models/HMM_Work/JHU_daily.csv", index_col=0)
#list of lists of deaths data
with open('NYT_daily_Warp_Death.txt') as f:
    NYT_daily_Warp_Death = simplejson.load(f)
with open('NYT_daily_Death_Filled.txt') as g:
    NYT_daily_Death_Filled = simplejson.load(g)
with open('JHU_daily_death.txt') as h:
    JHU_daily_death = simplejson.load(h)
#DTW Based Clusters
DTW_Clusters = pd.read_csv(f"{homedir}/models/HMM_Work/DTW_Clustering.csv", index_col=0)

JHU_Z_T_HMMs = makeHMMlist(JHU, DTW_Clusters, 'JHU_Z_T')
JHU_Z_L_HMMs = makeHMMlist(JHU, DTW_Clusters, 'JHU_Z_L')
JHU_N_T_HMMs = makeHMMlist(JHU, DTW_Clusters, 'JHU_N_T')
JHU_N_L_HMMs = makeHMMlist(JHU, DTW_Clusters, 'JHU_N_L')

NYT_F_Z_T_HMMs = makeHMMlist(NYT_F, DTW_Clusters, 'NYT_F_Z_T')
NYT_F_Z_L_HMMs = makeHMMlist(NYT_F, DTW_Clusters, 'NYT_F_Z_L')
NYT_F_N_T_HMMs = makeHMMlist(NYT_F, DTW_Clusters, 'NYT_F_N_T')
NYT_F_N_L_HMMs = makeHMMlist(NYT_F, DTW_Clusters, 'NYT_F_N_L')
NYT_F_N_L_L_HMMs = makeHMMlist(NYT_F, DTW_Clusters, 'NYT_F_N_L_L')

NYT_W_Z_T_HMMs = makeHMMlist(NYT_W, DTW_Clusters, 'NYT_W_Z_T')
NYT_W_Z_L_HMMs = makeHMMlist(NYT_W, DTW_Clusters, 'NYT_W_Z_L')
NYT_W_N_T_HMMs = makeHMMlist(NYT_W, DTW_Clusters, 'NYT_W_N_T')
NYT_W_N_L_HMMs = makeHMMlist(NYT_W, DTW_Clusters, 'NYT_W_N_L')

