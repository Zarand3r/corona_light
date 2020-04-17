import pandas as pd
import numpy as np
import os
import git
from pathlib import Path

repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
inputdir = f"{homedir}" + "/data/us/hospitals/"
outputdir = f"{homedir}" + "/models/data/us/health/"
Path(outputdir).mkdir(parents=True, exist_ok=True)
outputdir += "county_"

def main():
	County_Beds = pd.read_csv(inputdir+'beds_by_county.csv', encoding='latin1')
	County_ICU = pd.read_csv(inputdir+'icu_county.csv', encoding='latin1')

	#Cleaning beds Data
	County_Beds = County_Beds.sort_values(by=['FIPS'])
	#Dropping unecessary columns
	County_Beds = County_Beds.drop(columns=['state', 'county', 'Name']) 

	#Cleaning icu Data
	County_ICU = County_ICU.sort_values(by=['FIPS'])

	#Merging Hospital Data with ICU Data
	County_Health = County_ICU.join(County_Beds.set_index('FIPS'), on='FIPS', lsuffix='_Post', rsuffix='_Pre')
	#Cleaning Merged Dataset
	County_Health = County_Health[['FIPS','hospitals','staffed_beds','licensed_beds','icu_beds_Pre','icu_beds_Post']]
	#Removing NaNs
	County_Health = County_Health[County_Health.staffed_beds >= -1]
	County_Health = County_Health.sort_values(by=['FIPS'])
	County_Health = County_Health.set_index('FIPS')
	County_Health.to_csv(outputdir+'hospitals.csv')


	#Setting FIPS to main index
	County_Beds = County_Beds.set_index('FIPS')
	County_Beds.to_csv(outputdir+'beds.csv')

	County_ICU = County_ICU.set_index('FIPS')
	County_ICU.to_csv(outputdir+'ICU.csv')

if __name__ == '__main__':
	main()
