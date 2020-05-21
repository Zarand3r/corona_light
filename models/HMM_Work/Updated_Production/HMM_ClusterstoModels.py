#!/usr/bin/env python
# coding: utf-8

#Takes in clusterings and sends the fitted models to pickle

#File requires the ClusterMaker file to be run, so that 
#DTW_Clustering.csv, NYT_daily_Filled.csv, NYT_daily_Warped.csv,
#JHU_daily.csv, NYT_daily_Warp_Death.txt, NYT_daily_Death_Filled.txt
#and JHU_daily_death.txt are in the directory


import pandas as pd
import numpy as np
import os
import datetime
import simplejson
from hmmlearn import hmm
import pickle
import warnings
warnings.filterwarnings("error")
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

def makeX(Data, DTW, cluster_col, cluster_num, fipsname='FIPS', deathsname='Deaths'):
    #Takes in the dataset, cluster column and number, and gives out the deaths info in this cluster
    #In the form able to be processed by hmmlearn's HMM modules    
    fips = list(DTW[DTW[cluster_col] == cluster_num]['FIPS'])
    Rows = Data[Data[fipsname].isin(fips)]
    RawData = makeHMMUnSupData(Rows, deathsname, fipsname)
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
                    if scores[-1] > 0 and scores[-1] < scores[-2]:
                        Flag = False
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
        ind = np.argmax(np.diff(scores)[(n + 1):]/scores[(n + 2):])
        return hmm.GaussianHMM(n_components = ind + n + 3, covariance_type="full").fit(X[0],X[1])
    
def makeHMMlist(Data, DTW, cluster_col):
    labels = np.sort(DTW[cluster_col].dropna().unique())
    HMM_list = [0] * len(labels)
    n = 0
    for i in labels:
        X = makeX(Data, DTW, cluster_col, i)
        ls = [a.tolist()[0] for a in X[0]]
        HMM_list[n] = makeHMM(X)
        name = str(cluster_col) + str('_') + str(i) + str('.pkl')
        saveHMM(HMM_list[n], str(name))
        n += 1
    return None

def saveHMM(HMM, name):
    with open(name, "wb") as file: pickle.dump(HMM, file)

def main():
    #Dataframes of deaths
    NYT_F = pd.read_csv("NYT_daily_Filled.csv", index_col=0)
    NYT_W = pd.read_csv("NYT_daily_Warp.csv", index_col=0)
    NYT_F = NYT_F.rename(columns={'fips':'FIPS','deaths':'Deaths'})
    NYT_W = NYT_W.rename(columns={'fips':'FIPS','deaths':'Deaths'})
    JHU = pd.read_csv("JHU_daily.csv", index_col=0)
    #list of lists of deaths data
    with open('NYT_daily_Warp_Death.txt') as f:
        NYT_daily_Warp_Death = simplejson.load(f)
    with open('NYT_daily_Death_Filled.txt') as g:
        NYT_daily_Death_Filled = simplejson.load(g)
    with open('JHU_daily_death.txt') as h:
        JHU_daily_death = simplejson.load(h)
    #DTW Based Clusters
    DTW_Clusters = pd.read_csv("DTW_Clustering.csv", index_col=0)

    #Creation and saving of HMMs
    makeHMMlist(JHU, DTW_Clusters, 'JHU_Z_T')
    makeHMMlist(JHU, DTW_Clusters, 'JHU_Z_L')
    makeHMMlist(JHU, DTW_Clusters, 'JHU_N_T')
    makeHMMlist(JHU, DTW_Clusters, 'JHU_N_L')

    makeHMMlist(NYT_F, DTW_Clusters, 'NYT_F_Z')
    makeHMMlist(NYT_F, DTW_Clusters, 'NYT_F_N')

    makeHMMlist(NYT_W, DTW_Clusters, 'NYT_W_Z_T')
    makeHMMlist(NYT_W, DTW_Clusters, 'NYT_W_Z_L')
    makeHMMlist(NYT_W, DTW_Clusters, 'NYT_W_N_T')
    makeHMMlist(NYT_W, DTW_Clusters, 'NYT_W_N_L')

if __name__ == "__main__":
    main()

