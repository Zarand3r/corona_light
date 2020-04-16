import pandas as pd
import numpy as np
import os
import git
from pathlib import Path

repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
inputdir = f"{homedir}" + "/data/us/covid/"
outputdir = f"{homedir}" + "/models/data/us/covid/"
Path(outputdir).mkdir(parents=True, exist_ok=True)
outputdir += "county_"


def delete_zero_columns(frame): #Delete Columns containing only 0s
    cols = list(frame.columns)
    for col in cols:
        if (frame[col] == 0).all():
            frame = frame.drop(columns=[col])
    return frame        

def main():
	#This is the best record for cases/deaths out of all other sources, has most FIPS codes included
	USAFacts_C = pd.read_csv(inputdir+'confirmed_cases.csv')
	USAFacts_D = pd.read_csv(inputdir+'deaths.csv')

	#Less inclusive set of cases/deaths for counties
	JHU_CD = pd.read_csv(inputdir+'JHU_daily_US.csv')
	NYTCounties_CD = pd.read_csv(inputdir+'nyt_us_counties.csv')
	NYTCounties_CD_Daily = pd.read_csv(inputdir+'nyt_us_counties_daily.csv')

	#Removing Statewide Unallocated Lines   #Can be changed later
	USAFacts_C = USAFacts_C[USAFacts_C['County Name'] != 'Statewide Unallocated']
	USAFacts_D = USAFacts_D[USAFacts_D['County Name'] != 'Statewide Unallocated']

	#Merging Cases and Deaths
	USAFacts_CD = pd.merge(USAFacts_C, USAFacts_D, left_on='countyFIPS', right_on='countyFIPS', suffixes=('_C', '_D'))
	USAFacts_CD = USAFacts_CD.drop(columns=['County Name_D', 'State_D', 'stateFIPS_D'])
	USAFacts_CD = USAFacts_CD.drop(columns=['County Name_C', 'State_C', 'stateFIPS_C'])
	USAFacts_CD.columns = USAFacts_CD.columns.str.replace('countyFIPS','FIPS')

	USAFacts_CD_NonConsecutive = USAFacts_CD
	USAFacts_CD_NonConsecutive = USAFacts_CD_NonConsecutive.set_index('FIPS')
	USAFacts_CD_NonConsecutive.to_csv(outputdir+'USAFacts_CDNonConsecutive.csv') #csv of Fips, All Dates of Cases, All Dates of Deaths

	Beg = list(USAFacts_CD.columns[:4]) #First four entries
	End = list(np.sort(np.array(USAFacts_CD.columns[4:]))) #Every Date Sorted
	cols = list(Beg + End) #Ordered Columns

	USAFacts_CDConsecutive = USAFacts_CD[cols]
	USAFacts_CDConsecutive = USAFacts_CDConsecutive.set_index('FIPS')
	USAFacts_CDConsecutive.to_csv(outputdir+'USAFacts_CDConsecutive.csv') #Csv of FIps, each date and the number of cases & deaths


	JHU_CD = JHU_CD[JHU_CD.FIPS <= 60000]
	JHU_CD = pd.pivot_table(JHU_CD, values=['Confirmed', 'Deaths'], index=['FIPS'], columns=['Date'])
	JHU_CD = pd.DataFrame(JHU_CD)
	JHU_CD = JHU_CD.fillna(0)
	JHU_CD.to_csv(outputdir+'JHU_CD.csv')


	NYTCounties_CD = NYTCounties_CD.sort_values(by=['fips', 'date'])
	NYTCounties_CD = NYTCounties_CD[NYTCounties_CD.fips <= 95000]
	NYTCounties_CD = pd.pivot_table(NYTCounties_CD, values=['cases', 'deaths'], index=['fips'], columns=['date'])
	NYTCounties_CD = NYTCounties_CD.fillna(0)
	NYTCounties_CD.to_csv(outputdir+'NYT_CD.csv')

	NYTCounties_CD_Daily.sort_values(by=['fips', 'date'])
	NYTCounties_CD_Daily = pd.pivot_table(NYTCounties_CD_Daily, values=['cases', 'deaths'], index=['fips'], columns=['date'])
	NYTCounties_CD_Daily = NYTCounties_CD_Daily.fillna(0)
	NYTCounties_CD_Daily.to_csv(outputdir+'NYT_CD_Daily.csv')

if __name__ == '__main__':
	main()