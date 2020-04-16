import pandas as pd
import numpy as np
import os
import git
from pathlib import Path

repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
inputdir = f"{homedir}" + "/data/us/demographics/"
outputdir = f"{homedir}" + "/models/data/us/demographics/"
Path(outputdir).mkdir(parents=True, exist_ok=True)
outputdir += "county_"

#Changes prefixes of column names
def drop_prefix(self, prefix, replace = ''):
    self.columns = self.columns.str.replace(prefix, replace)
    return self

#Removes all duplicate columns from dataframes
def drop_dup_col(df):
    df = df.loc[:,~df.columns.duplicated()]
    return df

def main():
	#Loading in County Data
	#Datasets that are altered & Resaved as such
	Density = pd.read_csv(inputdir+'county_land_areas.csv', encoding='latin1')
	Age_Race = pd.read_csv(inputdir+'acs_2018.csv', encoding='latin1')
	County_Pop_Pop60 = pd.read_csv(inputdir+'county_populations.csv', encoding='latin1')
	Area_Houses = pd.read_csv(inputdir+'county_land_areas.csv', encoding='latin1')
	Educ = pd.read_csv(inputdir+'education.csv', encoding='latin1')
	Unemp = pd.read_csv(inputdir+'unemployment.csv', delimiter="\t")
	Poverty = pd.read_csv(inputdir+'poverty.csv', delimiter="\t")
	Votes = pd.read_csv(inputdir+'countypres_2000-2016.csv', encoding='latin1')

	#Key to map FIPs values to State and County Name
	Key = pd.read_csv(f"{homedir}"+'/data/us/processing_data/fips_key.csv', encoding='latin1')
	Key = Key.drop(columns=['MSA/PMSA NECMA']) #Dropping unecessary column
	Key = Key.set_index('FIPS')
	Key.to_csv('county_key.csv')

	#Datasets that are not altered
	Pop_60 = pd.read_csv(inputdir+'county_populations.csv', encoding='latin1')
	Pop_60 = Pop_60.set_index('FIPS')
	Pop_60.to_csv(outputdir+'Pop_60.csv')

	#Cleaning Voting Data
	Votes = Votes[Votes.party != 'green']
	Votes = Votes[Votes.party != 'republican'] #Removing unneeded rows
	Votes = Votes[Votes.candidate != 'Other']
	Votes = Votes[Votes.FIPS >= -1] #Removing NaN rows
	Votes = Votes.drop(columns=['state', 'state_po', 'county', 'office', 'candidate', 'version']) #removing uneeded columns
	Votes.insert(5, "Prop_Blue", Votes.candidatevotes/Votes.totalvotes, True) #Adding column of fraction of pop that vote dem.
	Votes = Votes.drop(columns=['candidatevotes', 'party'])    
	Votes = Votes.pivot(index= 'FIPS', columns = 'year') #making FIPS main index
	Votes.to_csv(outputdir+'votes.csv')
	#Removing the pivot aspect from the Votes Dataset
	Votes = pd.read_csv(outputdir+'votes.csv', encoding='latin1')
	Votes = Votes.drop([0,1])
	Votes.columns = ['FIPS', 'Total_Votes_2000', 'Total_Votes_2004', 'Total_Votes_2008', 'Total_Votes_2012', 'Total_Votes_2016', 'Frac_Dem_2000', 'Frac_Dem_2004', 'Frac_Dem_2008', 'Frac_Dem_2012', 'Frac_Dem_2016']
	Votes.FIPS = Votes.FIPS.astype(str).astype(float).astype(int) #Rewriting the columns names
	Votes = Votes.set_index('FIPS')
	Votes.to_csv(outputdir+'votes.csv')

	#Cleaning the Racial/Age Data
	Age_Race = Age_Race.sort_values(by=['FIPS'])
	#removing these percent/ratio values as these are poorly rounded, can be manually computed later
	Age_Race = Age_Race[Age_Race.columns.drop(list(Age_Race.filter(regex='Percent')))]
	Age_Race = Age_Race[Age_Race.columns.drop(list(Age_Race.filter(regex='ratio')))]
	#Dropping unecessary columns prefixes 
	Age_Race = drop_prefix(Age_Race, 'Estimate!!')
	Age_Race = drop_prefix(Age_Race, 'SEX AND AGE!!')
	Age_Race = drop_prefix(Age_Race, 'RACE!!')
	Age_Race = drop_prefix(Age_Race, 'Total population!!') #Changing column title names
	Age_Race = drop_prefix(Age_Race, 'One race!!', 'Exclusively ')
	Age_Race = drop_prefix(Age_Race, 'Two or more races!!', 'Interracial ')
	Age_Race = drop_prefix(Age_Race, 'Race alone or in combination with one or more other races!!', 'Total ')
	Age_Race = drop_prefix(Age_Race, 'HISPANIC OR LATINO AND ')
	#Dropping unecessary columns
	Age_Race = drop_dup_col(Age_Race) #Removes duplicate columns
	Age_Race = Age_Race[Age_Race.columns.drop(list(Age_Race.filter(regex='.1')))] #removes extra duplicate columns
	Age_Race = Age_Race.drop(columns=['Geographic Area Name', 'Total Total population'])
	Age_Race = Age_Race.replace('N', 0) #changing NaN values to 0
	#####################################

	Age_Race = Age_Race.set_index('FIPS')
	Age_Race.to_csv(outputdir+'age_race.csv')

	#Cleaning Education Data, removing state data from county data 
	Educ_County = Educ[Educ['FIPS'] % 1000 != 0]
	Educ_County = Educ_County.set_index('FIPS')
	Educ_County.to_csv(outputdir+'edu.csv')

	#Cleaning Density area Data
	Density = Density.drop(columns=['County Name']) #Dropping unecessary column
	Density.columns = Density.columns.str.replace('County FIPS','FIPS')
	Density = Density.set_index('FIPS')
	Density.to_csv(outputdir+'density.csv')

	#Cleaning Unemployment area Data
	Unemp = Unemp.drop(columns=['State', 'Area_name'])  #Dropping unecessary columns
	Unemp = Unemp.set_index('FIPS')
	Unemp.to_csv(outputdir+'unemp.csv')

	#Cleaning Poverty area Data
	Poverty = Poverty.drop(columns=['Stabr', 'Area_name', 'Rural-urban_Continuum_Code_2013', 'Urban_Influence_Code_2013']) 
	#Dropping unecessary columns
	Poverty = Poverty.set_index('FIPS')
	Poverty.to_csv(outputdir+'poverty.csv')


if __name__ == '__main__':
	main()
