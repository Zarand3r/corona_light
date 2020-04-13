import pandas as pd
import numpy as np
import os
import git
from pathlib import Path

repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
inputdir = f"{homedir}" + "/data/us/other/"
outputdir = f"{homedir}" + "/models/data/us/other/"
Path(outputdir).mkdir(parents=True, exist_ok=True)
outputdir += "county_"


def main():
	#Loading in data
	Air_Traffic = pd.read_csv(inputdir+'air_traffic.csv', encoding='latin1')
	Policies = pd.read_csv(inputdir+'policies.csv', encoding='latin1')
	Transit = pd.read_csv(inputdir+'transit.csv', encoding='latin1')

	#Separating only county rows
	Policies_County = Policies[Policies.FIPS%1000 != 0]
	#Removing extra columns
	Policies_County = Policies_County.drop(columns = ['STATE', 'AREA_NAME'])
	Policies_County = Policies_County.set_index('FIPS')
	Policies_County.to_csv(outputdir+'policy.csv')

	#Separating only cleaning Transit Data
	Transit = Transit.sort_values(by=['FIPS'])
	Transit = Transit.drop(columns = ['Name', 'Population'])
	Transit = Transit.set_index('FIPS')
	Transit.to_csv(outputdir+'transit.csv')

	#Merging & Cleaning Policy/Transit Data
	Policy_Transit = pd.merge(Policies_County, Transit, left_on='FIPS', right_on='FIPS')
	Policy_Transit.to_csv(outputdir+'policy_transit.csv')

if __name__ == '__main__':
	main()

