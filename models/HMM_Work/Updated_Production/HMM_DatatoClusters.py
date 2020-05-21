#!/usr/bin/env python
# coding: utf-8

#Takes in raw data, outputs clusters based on DTW

#Number of clusters for dataset depends on the arrays 
#JHU_Cluster_Size, NYT_F_Cluster_Size, and NYT_W_Cluster_Size

import pandas as pd
import numpy as np
import os
import datetime
import simplejson
from dtaidistance import dtw
from dtaidistance import clustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

def makeHMMUnSupData(Input, colname, fipsname):
    #Takes input dataframe, and gives out HMM format of Input data, a list of lists 
    #of the colname value, each list in the set represents one fips code.
    Output = []
    for fips in Input[fipsname].unique():
        temp = list(Input[Input[fipsname] == fips][colname])
        Output.append(temp)
    return Output

def monotonicCol(Data, colname):
    #Takes a column that should have monotonically increasing data for a column (number of deaths)
    #and adjusts the column to ensure this property, iterating backwards through each fips code's entries
    ls = []
    tempvals = []
    for fips in Data.FIPS.unique():
        vals = list(Data[Data['FIPS'] == fips][colname])
        flag = True
        for val in reversed(vals):
            if flag:
                flag = False
                maxval = val
                tempvals.append(maxval)
            else:
                if val > maxval:
                    tempvals.append(maxval)
                else:
                    maxval = val
                    tempvals.append(val)
        ls.extend(reversed(tempvals))
        tempvals = []
    return ls

def cumtoDaily(Data, colname):
    #Takes cumulative column data and turns the data into daily changes 
    ls = []
    column = Data[colname]
    for fips in Data.FIPS.unique():
        ls.extend(list(Data[Data['FIPS'] == fips][colname].diff().fillna(0)))
    return ls

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

def main():
    #NYT Data (NYT_W and NYT_F)
    #Differenced Daily Death Data
    NYT_daily = pd.read_csv(f"{homedir}/data/us/covid/nyt_us_counties_daily.csv")
    NYT_daily = NYT_daily.drop(columns=['county','state']).sort_values(['fips','date']).reset_index(drop=True)
    NYT_daily['fips'] = NYT_daily.fips.astype(int)
    NYT_daily['date'] = pd.to_datetime(NYT_daily['date'])
    NYT_daily['id'] = NYT_daily.fips.astype(str).str.cat(NYT_daily.date.astype(str), sep=', ')
    FirstDay = min(NYT_daily.date.unique())
    LastDay = max(NYT_daily.date.unique())

    #Making a time-warping of NYT daily data, so each county has a value at the starting day of 2020-01-21, the second value is
    #the date of the first reported date from NYT
    # and then a final value at the most recent day
    NYT_daily_Warp = NYT_daily
    for fips in NYT_daily.fips.unique():
        rows = NYT_daily[NYT_daily['fips'] == fips]
        #adding in the first day values
        if FirstDay not in rows.date.unique():
            NYT_daily_Warp = NYT_daily_Warp.append({'fips': fips, 'date': pd.to_datetime('2020-01-21'), 'cases': 0, 'deaths' : 0, 'id' : str(fips) + ', 2020-01-21'}, ignore_index=True)
        #making sure each entry has the final day values
        if LastDay not in rows.date.unique():
            NYT_daily_Warp = NYT_daily_Warp[NYT_daily_Warp['fips'] != fips]
    NYT_daily_Warp = NYT_daily_Warp.sort_values(['fips','date']).reset_index(drop=True)
    NYT_daily_Warp.to_csv('NYT_daily_Warp.csv')
    NYT_daily_Warp_Death = makeHMMUnSupData(NYT_daily_Warp, 'deaths', 'fips')

    #This is a list of all the counties and dates
    County_List = list(NYT_daily.fips.unique())
    Date_List = list(NYT_daily.date.unique())
    #This creates a base dataframe that contains all pairs of FIPS codes with the valid dates given in Air_Qual
    CL, DL = pd.core.reshape.util.cartesian_product([County_List, Date_List])
    BaseFrame = pd.DataFrame(dict(fips=CL, date=DL)).sort_values(['fips','date']).reset_index(drop=True)
    BaseFrame['id'] = BaseFrame.fips.astype(str).str.cat(BaseFrame.date.astype(str), sep=', ')

    #Making frame of all deaths at all dates to properly do DTW clustering
    NYT_daily_Filled = BaseFrame.join(NYT_daily.set_index('id'), on='id', how='outer', lsuffix='',rsuffix='_x').sort_values(['fips', 'date']).drop(columns=['fips_x','date_x']).fillna(0).drop_duplicates(subset=['fips','date']).reset_index(drop=True)
    NYT_daily_Filled.to_csv('NYT_daily_Filled.csv')
    #List of lists of daily death count for each county, starting 1/23/20, ending most recent date.
    NYT_daily_Death_Filled = makeHMMUnSupData(NYT_daily_Filled, 'deaths', 'fips')
    #JHU Data
    JHU_tot = pd.read_csv(f"{homedir}/data/us/covid/JHU_daily_US.csv").sort_values(['FIPS','Date'])
    FIPSlist = JHU_tot.FIPS.unique()
    Datelist = JHU_tot.Date.unique()
    Datepair = [Datelist[0],Datelist[-1]]

    #Getting rid of unneded fips code in the list of total codes
    for fips in FIPSlist:
        rows = JHU_tot[JHU_tot['FIPS'] == fips]
        datelist = rows.Date.unique()
        datepair = [datelist[0],datelist[-1]]
        if np.array_equal(Datepair,datepair) != True:
            JHU_tot = JHU_tot.drop(list(JHU_tot[JHU_tot['FIPS'] == fips].index))
    JHU_tot = JHU_tot.sort_values(['FIPS','Date']).reset_index(drop=True)

    d = {'FIPS': JHU_tot['FIPS'], 'Date' : JHU_tot['Date'], 'Confirmed' : monotonicCol(JHU_tot,'Confirmed'),'Deaths' : monotonicCol(JHU_tot,'Deaths'),'Active' : monotonicCol(JHU_tot,'Active'),'Recovered' : monotonicCol(JHU_tot,'Recovered')}
    #Monotonically increaasing transformation of JHU_tot
    JHU_mono = pd.DataFrame(data=d)

    d = {'FIPS': JHU_mono['FIPS'], 'Date' : JHU_mono['Date'], 'Confirmed' : cumtoDaily(JHU_mono,'Confirmed'),'Deaths' : cumtoDaily(JHU_mono,'Deaths'),'Active': cumtoDaily(JHU_mono,'Active'),'Recovered' : cumtoDaily(JHU_mono,'Recovered')}
    #Daily changing data based on monotonically transformed data
    JHU_daily = pd.DataFrame(data=d)
    JHU_daily.to_csv('JHU_Daily.csv')
    #List of lists of daily death count for each county, starting 3/23/20, ending most recent date.
    JHU_daily_death = makeHMMUnSupData(JHU_daily, 'Deaths', 'FIPS')
    
    NYT_F = NYT_daily_Filled
    NYT_W = NYT_daily_Warp
    JHU = JHU_daily
    
    #Original dataset, making into list of np arrays
    NYT_daily_Warp_Death = [np.array(x) for x in NYT_daily_Warp_Death]
    NYT_daily_Death_Filled = [np.array(x) for x in NYT_daily_Death_Filled]
    JHU_daily_death = [np.array(x) for x in JHU_daily_death]
    
    #Saving the death data files
    f = open('NYT_daily_Warp_Death.txt', 'w')
    simplejson.dump(NYT_daily_Warp_Death, f)
    f.close()
    g = open('NYT_daily_Death_Filled.txt', 'w')
    simplejson.dump(NYT_daily_Death_Filled, g)
    g.close()
    h = open('JHU_daily_death.txt', 'w')
    simplejson.dump(JHU_daily_death, h)
    h.close()

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
    JHU_Cluster_Size = [2,3,6,3,6]

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

    NYT_F_Cluster_Size = [2,5,5]

    Z_NYT_F_O = makeZ(NYT_daily_Death_Filled)
    Z_NYT_F_Z = makeZ(Series_NYT_F)
    Z_NYT_F_N = makeZ(Series_NYT_F_nozeros)

    NYT_F_O = fcluster(Z_NYT_F_O, NYT_F_Cluster_Size[0], criterion ='maxclust')
    NYT_F_Z = fcluster(Z_NYT_F_Z, NYT_F_Cluster_Size[1], criterion ='maxclust')
    NYT_F_N = fillnonzero(Series_NYT_F,fcluster(Z_NYT_F_N, NYT_F_Cluster_Size[2], criterion ='maxclust'))

    ClustersNYT_F = pd.DataFrame(data=NYT_F.fips.unique(),columns=['FIPS'])
    ClustersNYT_F['NYT_F_Orig'] = NYT_F_O
    ClustersNYT_F['NYT_F_Z'] = NYT_F_Z
    ClustersNYT_F['NYT_F_N'] = NYT_F_N

    NYT_W_Cluster_Size = [2,5,8,5,7]

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
    
    #Saving all the clusters in one dataframe
    AllClusters = ClustersJHU.join(ClustersNYT_F.set_index('FIPS'), on='FIPS', how='outer').join(ClustersNYT_W.set_index('FIPS'), on='FIPS', how='outer').sort_values('FIPS')
    AllClusters.to_csv('DTW_Clustering.csv')

if __name__ == "__main__":
    main()

