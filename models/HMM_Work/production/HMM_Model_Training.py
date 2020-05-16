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

#Dataframes of deaths
NYT_F = pd.read_csv(f"{homedir}/models/HMM_Work/NYT_daily_Filled.csv", index_col=0)
NYT_W = pd.read_csv(f"{homedir}/models/HMM_Work/NYT_daily_Warp.csv", index_col=0)
JHU = pd.read_csv(f"{homedir}/models/HMM_Work/JHU_daily.csv", index_col=0)
#list of lists of deaths data
with open('NYT_daily_Warp_Death.txt') as f:
    NYT_daily_Warp_Death = json.load(f)
with open('NYT_daily_Death_Filled.txt') as g:
    NYT_daily_Death_Filled = json.load(g)
with open('JHU_daily_death.txt') as h:
    JHU_daily_death = json.load(h)
#DTW Based Clusters
DTW_Clusters = pd.read_csv(f"{homedir}/models/HMM_Work/DTW_Clustering.csv", index_col=0)

#Training data for the models
test = makeX(NYT_F, DTW_Clusters, 'NYT_W_Z_L', 3, 'fips', "deaths")

#A bunch of models

model1 = hmm.GaussianHMM(n_components=4, covariance_type="full")
model2 = hmm.GMMHMM(n_components=4, n_mix=1, covariance_type="full")
model3 = hmm.GaussianHMM(n_components=10, covariance_type="full")
# model4 = hmm.GMMHMM(n_components=10, n_mix=2, covariance_type="full")
# model5 = hmm.GaussianHMM(n_components=20, covariance_type="full")
# model6 = hmm.GMMHMM(n_components=15, n_mix=3, covariance_type="full")
# model7 = hmm.GaussianHMM(n_components=4, covariance_type="full", algorithm='map')
# model8 = hmm.GMMHMM(n_components=4, n_mix=2, covariance_type="full", algorithm='map')
# model9 = hmm.GaussianHMM(n_components=10, covariance_type="full", algorithm='map')
# model10 = hmm.GMMHMM(n_components=10, n_mix=2, covariance_type="full", algorithm='map')
# model11 = hmm.GaussianHMM(n_components=20, covariance_type="full", algorithm='map')
# model12 = hmm.GMMHMM(n_components=15, n_mix=3, covariance_type="full", algorithm='map')

model1.fit(test[0],test[1])
model2.fit(test[0],test[1])
model3.fit(test[0],test[1])


model3 = hmm.GaussianHMM(n_components=4, covariance_type="full")
model3.fit(test[0],test[1])
model3.score(test[0],test[1])


model1.get_stationary_distribution()
model1.score_samples(test[0],test[1])

X1 = [0.5, 1.0, -1.0, 0.42, 0.24]
X2 = [2.4, 4.2, 0.5, -0.24]
X = np.concatenate([X1, X2]).reshape(-1,1)
lengths = [len(X1), len(X2)]
hmm.GaussianHMM(n_components=2).fit(X, lengths)

#This is just a testing file so far, because our actual HMM clusterings are not available
#Maknig basic list of list data from the direct NYT Data (no clustering we just take the whole dataset)
DailyDeathUnSup = makeHMMUnSupData(NYT_daily, 'deaths', 'fips')
#Making the mapping of number of deaths to HMM states
[DailyDeathMap, DailyDeathRMap, DailyDeathUnSupHMM] = makeHMMmap(DailyDeathUnSup)
#Making supervised X and Y datasets
DailyDeathSup = makeHMMSupData(DailyDeathUnSupHMM)

#using the superviesed testing data, and making a supervised HMM from this 
SupHMM = supervised_HMM(DailyDeathSup[0],DailyDeathSup[1])
SupHMM

test = np.zeros(14)
for j in range(1000): #This generates a sample of length 14, with a starting state of 2, 
    test += np.array(sample_sentence(SupHMM, DailyDeathMap, 14, 2))#state of 2 means 2 people died in the county yesterday
print(test/1000)




