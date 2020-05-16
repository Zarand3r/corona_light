#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import datetime
from HMM import unsupervised_HMM
from HMM import supervised_HMM
from HMM_helper import sample_sentence


# In[ ]:


import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir


# In[ ]:


def makeHMMUnSupData(Input, colname, fipsname):
    #Takes input dataframe, and gives out HMM format of data, a list of lists 
    #of the colname value, each list in the set represents one fips code.
    Output = []
    for fips in Input[fipsname].unique():
        temp = list(Input[Input[fipsname] == fips][colname])
        Output.append(temp)
    return Output


# In[ ]:


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


# In[ ]:


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


# In[2]:


#Differenced Daily Death Data
NYT_daily = pd.read_csv(f"{homedir}/data/us/covid/nyt_us_counties_daily.csv")
NYT_daily = NYT_daily.drop(columns=['county','state']).sort_values(['fips','date']).reset_index(drop=True)
NYT_daily['fips'] = NYT_daily.fips.astype(int)
NYT_daily['date'] = pd.to_datetime(NYT_daily['date'])
NYT_daily['id'] = NYT_daily.fips.astype(str).str.cat(NYT_daily.date.astype(str), sep=', ')


# In[10]:


#This is just a testing file so far, because our actual HMM clusterings are not available
#Maknig basic list of list data from the direct NYT Data (no clustering we just take the whole dataset)
DailyDeathUnSup = makeHMMUnSupData(NYT_daily, 'deaths', 'fips')
#Making the mapping of number of deaths to HMM states
[DailyDeathMap, DailyDeathRMap, DailyDeathUnSupHMM] = makeHMMmap(DailyDeathUnSup)
#Making supervised X and Y datasets
DailyDeathSup = makeHMMSupData(DailyDeathUnSupHMM)


# In[11]:


#using the superviesed testing data, and making a supervised HMM from this 
SupHMM = supervised_HMM(DailyDeathSup[0],DailyDeathSup[1])
SupHMM


# In[13]:


test = np.zeros(14)
for j in range(1000): #This generates a sample of length 14, with a starting state of 2, 
    test += np.array(sample_sentence(SupHMM, DailyDeathMap, 14, 2))#state of 2 means 2 people died in the county yesterday
print(test/1000)


# In[ ]:




