#!/usr/bin/env python
# coding: utf-8

#Takes in HMM models and sends the predicted outputs to csv.

#File requires the DatatoClusters file to be run, so that 
#DTW_Clustering.csv, NYT_daily_Filled.csv, NYT_daily_Warped.csv,
#JHU_daily.csv, NYT_daily_Warp_Death.txt, NYT_daily_Death_Filled.txt
#and JHU_daily_death.txt are in the directory
#File also requires ClusterstoModels to be run,
#So all the pickled HMMs are in directory

import pandas as pd
import numpy as np
import os
from hmmlearn import hmm
from sklearn.utils import check_random_state
import pickle
import simplejson
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

def makeHMMUnSupData(Input, colname='Deaths', fipsname='FIPS'):
    #Takes input dataframe, and gives out HMM format of data, a list of lists 
    #of the colname value, each list in the set represents one fips code.
    Output = []
    for fips in Input[fipsname].unique():
        temp = list(Input[Input[fipsname] == fips][colname])
        Output.append(temp)
    return Output

def loadHMM(name):
    with open(name, "rb") as file: x = pickle.load(file)
    return x

def loadHMMlist(colname, minclust, maxclust):
    HMMlist = [0]*(maxclust - minclust + 1)
    n = 0
    labels = [0]*(maxclust - minclust + 1)
    for i in range(minclust, maxclust + 1):
        name = str(colname) + '_' + str(i) + '.0.pkl'
        HMMlist[n] = loadHMM(str(name))
        labels[n] = i
        n += 1
    return HMMlist, labels

def makeFipsPrediction(HMM, Data, fipscode, length=14, n_iters=10):
    #Takes in an HMM, a dataset (either JHU, NYT_F, or NYT_W) and a fips code,
    #Gives the HMM state predictions and emission predictions
    #Does this predictions n_iters times, and reports the average states/emissions
    X = makeHMMUnSupData(Data[Data['FIPS']==fipscode])[0]
    states = HMM.predict(np.array(X).reshape(-1,1))
    transmat_cdf = np.cumsum(HMM.transmat_, axis=1)
    Emissions = [0.0] * length
    States = [0.0] * length
    
    for i in range(n_iters):
        for j in range(length):
            random_state = check_random_state(HMM.random_state)
            if j == 0:
                next_state = (transmat_cdf[states[-1]] > random_state.rand()).argmax()
            else:
                next_state = (transmat_cdf[next_state] > random_state.rand()).argmax()
            
            next_obs = HMM._generate_sample_from_state(next_state, random_state)
            
            Emissions[j] += next_obs[0]/n_iters
            States[j] += next_state/n_iters
            
    return States, Emissions

def makeHMMListPrediction(HMMList, Data, colname, DTW, length=14, n_iters=10):
    HMMs = HMMList[0]
    labels = HMMList[1]
    PredictionFrame = DTW[~DTW[colname].isna()][['FIPS']]
    
    for i in range(length):
        PredictionFrame[str(1 + i)] = 0
    n = 0
    
    for i in labels:
        codes = DTW[DTW[colname] == i]['FIPS'].unique().tolist()
        HMM = HMMs[n]
        for code in codes:
            Prediction = makeFipsPrediction(HMM, Data, code, length, n_iters)[1]
            for j in range(length):
                PredictionFrame.loc[PredictionFrame['FIPS'] == code, str(j + 1)] = Prediction[j]
        n += 1
    return PredictionFrame

def main(num_iterations=10):
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
    
    #Loading in HMMs
    JHU_Z_T_HMMs = loadHMMlist('JHU_Z_T', 1, 3)
    JHU_Z_L_HMMs = loadHMMlist('JHU_Z_L', 1, 6)
    JHU_N_T_HMMs = loadHMMlist('JHU_N_T', 0, 3)
    JHU_N_L_HMMs = loadHMMlist('JHU_N_L', 0, 6)

    #Making predictions
    JHU_Z_T_Pred = makeHMMListPrediction(JHU_Z_T_HMMs, JHU, 'JHU_Z_T', DTW_Clusters, length=14, n_iters=num_iterations)
    JHU_Z_L_Pred = makeHMMListPrediction(JHU_Z_L_HMMs, JHU, 'JHU_Z_L', DTW_Clusters, length=14, n_iters=num_iterations)
    JHU_N_T_Pred = makeHMMListPrediction(JHU_N_T_HMMs, JHU, 'JHU_N_T', DTW_Clusters, length=14, n_iters=num_iterations)
    JHU_N_L_Pred = makeHMMListPrediction(JHU_N_L_HMMs, JHU, 'JHU_N_L', DTW_Clusters, length=14, n_iters=num_iterations)

    JHU_Z_T_Pred.to_csv('JHU_Z_T_Pred.csv')
    JHU_Z_L_Pred.to_csv('JHU_Z_L_Pred.csv')
    JHU_N_T_Pred.to_csv('JHU_N_T_Pred.csv')
    JHU_N_L_Pred.to_csv('JHU_N_L_Pred.csv')

    NYT_F_Z_HMMs = loadHMMlist('NYT_F_Z', 1, 5)
    NYT_F_N_HMMs = loadHMMlist('NYT_F_N', 0, 5)

    NYT_F_Z_Pred = makeHMMListPrediction(NYT_F_Z_HMMs, NYT_F, 'NYT_F_Z', DTW_Clusters, length=14, n_iters=num_iterations)
    NYT_F_N_Pred = makeHMMListPrediction(NYT_F_N_HMMs, NYT_F, 'NYT_F_N', DTW_Clusters, length=14, n_iters=num_iterations)

    NYT_F_Z_Pred.to_csv('NYT_F_Z_Pred.csv')
    NYT_F_N_Pred.to_csv('NYT_F_N_Pred.csv')

    NYT_W_Z_T_HMMs = loadHMMlist('NYT_W_Z_T', 1, 5)
    NYT_W_Z_L_HMMs = loadHMMlist('NYT_W_Z_L', 1, 8)
    NYT_W_N_T_HMMs = loadHMMlist('NYT_W_N_T', 0, 5)
    NYT_W_N_L_HMMs = loadHMMlist('NYT_W_N_L', 0, 7)

    NYT_W_Z_T_Pred = makeHMMListPrediction(NYT_W_Z_T_HMMs, NYT_W, 'NYT_W_Z_T', DTW_Clusters, length=14, n_iters=num_iterations)
    NYT_W_Z_L_Pred = makeHMMListPrediction(NYT_W_Z_L_HMMs, NYT_W, 'NYT_W_Z_L', DTW_Clusters, length=14, n_iters=num_iterations)
    NYT_W_N_T_Pred = makeHMMListPrediction(NYT_W_N_T_HMMs, NYT_W, 'NYT_W_N_T', DTW_Clusters, length=14, n_iters=num_iterations)
    NYT_W_N_L_Pred = makeHMMListPrediction(NYT_W_N_L_HMMs, NYT_W, 'NYT_W_N_L', DTW_Clusters, length=14, n_iters=num_iterations)

    NYT_W_Z_T_Pred.to_csv('NYT_W_Z_T_Pred.csv')
    NYT_W_Z_L_Pred.to_csv('NYT_W_Z_L_Pred.csv')
    NYT_W_N_T_Pred.to_csv('NYT_W_N_T_Pred.csv')
    NYT_W_N_L_Pred.to_csv('NYT_W_N_L_Pred.csv')

if __name__ == "__main__":
    main(25)

