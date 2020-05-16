#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import os
import datetime
import json
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


#Cumulative Death Data
NYT_tot = pd.read_csv(f"{homedir}/data/us/covid/nyt_us_counties.csv")
NYT_tot.loc[NYT_tot["county"]=='New York City', "fips"]=36061
NYT_tot = NYT_tot.drop(columns=['county','state']).sort_values(['fips','date']).reset_index(drop=True)
NYT_tot = NYT_tot.dropna(subset=['fips'])
NYT_tot['fips'] = NYT_tot.fips.astype(int)
NYT_tot['date'] = pd.to_datetime(NYT_tot['date'])
NYT_tot['id'] = NYT_tot.fips.astype(str).str.cat(NYT_tot.date.astype(str), sep=', ')
#Making new parameter for deathrate
NYT_tot['deathrate'] = NYT_tot['deaths']/NYT_tot['cases']
NYT_tot = NYT_tot.fillna(0)
#multiplying death rate by 1000 to give integer state values
NYT_tot['deathstate'] = NYT_tot['deathrate']*1000
NYT_tot['deathstate'] = NYT_tot['deathstate'].astype(int)


#Differenced Daily Death Data
NYT_daily = pd.read_csv(f"{homedir}/data/us/covid/nyt_us_counties_daily.csv")
NYT_daily.loc[NYT_daily["county"]=='New York City', "fips"]=36061
NYT_daily = NYT_daily.drop(columns=['county','state']).sort_values(['fips','date']).reset_index(drop=True)
NYT_daily['fips'] = NYT_daily.fips.astype(int)
NYT_daily['date'] = pd.to_datetime(NYT_daily['date'])
NYT_daily['id'] = NYT_daily.fips.astype(str).str.cat(NYT_daily.date.astype(str), sep=', ')
FirstDay = min(NYT_daily.date.unique())
LastDay = max(NYT_daily.date.unique())




#Making a time-warping of NYT daily data, so each county has a value at the starting day of 2020-01-21
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


d = {'FIPS': JHU_tot['FIPS'], 'Date' : JHU_tot['Date'], 'Confirmed' : monotonicCol(JHU_tot,'Confirmed'), 'Deaths' : monotonicCol(JHU_tot,'Deaths'),'Active' : monotonicCol(JHU_tot,'Active'), 'Recovered' : monotonicCol(JHU_tot,'Recovered')}
JHU_mono = pd.DataFrame(data=d)
d = {'FIPS': JHU_mono['FIPS'], 'Date' : JHU_mono['Date'], 'Confirmed' : cumtoDaily(JHU_mono,'Confirmed'), 'Deaths' : cumtoDaily(JHU_mono,'Deaths'),'Active' : cumtoDaily(JHU_mono,'Active'), 'Recovered' : cumtoDaily(JHU_mono,'Recovered')}
#Daily changing data based on monotonically transformed data
JHU_daily = pd.DataFrame(data=d)
JHU_daily.to_csv('JHU_Daily.csv')
#List of lists of daily death count for each county, starting 3/23/20, ending most recent date.
JHU_daily_death = makeHMMUnSupData(JHU_daily, 'Deaths', 'FIPS')

#Saving the death data filesw
f = open('NYT_daily_Warp_Death.txt', 'w')
json.dump(NYT_daily_Warp_Death, f)
f.close()

g = open('NYT_daily_Death_Filled.txt', 'w')
json.dump(NYT_daily_Death_Filled, g)
g.close()

h = open('JHU_daily_death.txt', 'w')
json.dump(JHU_daily_death, h)
h.close()



