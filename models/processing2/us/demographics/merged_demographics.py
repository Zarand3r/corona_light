import pandas as pd
import numpy as np
import os
import git


repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
inputdir = f"{homedir}" + "/data/us/demographics/"
outputdir = f"{homedir}" + "/models/data/us/demographics"

#Loading in County Data
Age_Race = pd.read_csv(inputdir+'acs_2018.csv', encoding='latin1')
County_Pop_Pop60 = pd.read_csv(inputdir+'county_populations.csv', encoding='latin1')
Area_Houses = pd.read_csv(inputdir+'county_land_areas.csv', encoding='latin1')
Educ = pd.read_csv(inputdir+'education.csv', encoding='latin1')
Unemp = pd.read_csv(inputdir+'unemployment.csv', delimiter="\t")
Poverty = pd.read_csv(inputdir+'poverty.csv', delimiter="\t")
Votes = pd.read_csv(inputdir+'countypres_2000-2016.csv', encoding='latin1')

#State Populations
State_Pop = pd.read_csv(inputdir+'state_populations.csv', encoding='latin1')


#Changes prefixes of column names
def drop_prefix(self, prefix, replace = ''):
    self.columns = self.columns.str.replace(prefix, replace)
    return self


#Cleaning Voting Data
Votes = Votes[Votes.party != 'green']
Votes = Votes[Votes.party != 'republican'] #Removing unneeded rows
Votes = Votes[Votes.candidate != 'Other']
Votes = Votes[Votes.FIPS <= 10000000]

Votes = Votes.drop(columns=['state', 'state_po', 'county', 'office', 'candidate', 'version']) #removing uneeded columns

Votes.insert(5, "Prop_Blue", Votes.candidatevotes/Votes.totalvotes, True) #Adding column of proportion of democratic
Votes = Votes.drop(columns=['candidatevotes', 'party'])    
Votes = Votes.pivot(index= 'FIPS', columns = 'year') #making FIPS main index
Votes.to_csv(outputdir+'Votes.csv')


#Cleaning the Racial/Age Data
Age_Race.sort_values(by=['FIPS'])
Age_Race = Age_Race[Age_Race.columns.drop(list(Age_Race.filter(regex='Percent')))]
Age_Race = Age_Race[Age_Race.columns.drop(list(Age_Race.filter(regex='ratio')))]
#Dropping unecessary columns
Age_Race = drop_prefix(Age_Race, 'Estimate!!')
Age_Race = drop_prefix(Age_Race, 'SEX AND AGE!!')
Age_Race = drop_prefix(Age_Race, 'RACE!!')
Age_Race = drop_prefix(Age_Race, 'Total population!!') #Changing column title names
Age_Race = drop_prefix(Age_Race, 'One race!!', 'Race_1 ')
Age_Race = drop_prefix(Age_Race, 'Two or more races!!', 'Race_2+ ')
Age_Race = drop_prefix(Age_Race, 'Race alone or in combination with one or more other races!!', 'Race_Total ')
Age_Race = drop_prefix(Age_Race, 'Race alone or in combination with one or more other races!!', 'Race_Total ')
Age_Race = drop_prefix(Age_Race, 'HISPANIC OR LATINO AND', 'Hispanic:')
Age_Race.to_csv(outputdir+'Age_Race.csv')


# #Joining Population and Unemployment Data
# Pop_Unemp =  pd.merge(County_Pop_Pop60, Unemp, left_on='FIPS', right_on='FIPS')
# #Joining this with Land Area/Density Data
# Pop_Unemp_Area = pd.merge(Pop_Unemp, Area_Houses, left_on='FIPS', right_on='County FIPS')
# Pop_Unemp_Area = Pop_Unemp_Area.drop(columns=['County FIPS', 'Area in square miles - Total area', 'County Name'])
# #Joining this with Education Data
# Pop_Unemp_Area_Educ = pd.merge(Pop_Unemp_Area, Educ, left_on='FIPS', right_on='FIPS')
# #Joining this with Poverty Data
# Pop_Unemp_Area_Educ_Pov = pd.merge(Pop_Unemp_Area_Educ, Poverty, left_on='FIPS', right_on='FIPS')
# #Joining this with Voting Data
# Pop_Unemp_Area_Educ_Pov_Votes = pd.merge(Pop_Unemp_Area_Educ_Pov, Votes, left_on='FIPS', right_on='FIPS')


# #Joining Compiled Data with Racial/Age Data (This is incomplete, not all FIPS included)
# Demographics_Full = pd.merge(Pop_Unemp_Area_Educ, Age_Race, left_on='FIPS', right_on='FIPS')

# pd.set_option('max_columns', None)

# #Repeating similar process to above, but changing order of merges to make maximum rows
# #Joining Education & Area
# Educ_Area = pd.merge(Educ, Area_Houses, left_on='FIPS', right_on='County FIPS')
# Educ_Area = Educ_Area.drop(columns=['County FIPS', 'Area in square miles - Total area', 'County Name'])
# #Joining this with Population Data
# Educ_Area_Pop = pd.merge(Educ_Area, County_Pop_Pop60, left_on='FIPS', right_on='FIPS')
# #Joining this with Unemployment Data
# Educ_Area_Pop_Unemp = pd.merge(Educ_Area_Pop, Unemp, left_on='FIPS', right_on='FIPS')
# #Joining this with Poverty Data
# Educ_Area_Pop_Unemp_Pov = pd.merge(Educ_Area_Pop_Unemp, Poverty, left_on='FIPS', right_on='FIPS')
# #Joining this with Voting Data
# Educ_Area_Pop_Unemp_Pov_Votes = pd.merge(Pop_Unemp_Area_Educ_Pov, Votes, left_on='FIPS', right_on='FIPS')




#####==== MERGING ====####

#Cleaning Votes
Votes = pd.read_csv(outputdir + "Votes.csv", encoding='latin1')
Votes = Votes.drop([0,1])
Votes.columns = ['FIPS', 'Total_2000', 'Total_2004', 'Total_2008', 'Total_2012', 'Total_2016', 'Dem_2000', 'Dem_2004', 'Dem_2008', 'Dem_2012', 'Dem_2016']
Votes.FIPS = Votes.FIPS.astype(str).astype(float).astype(int)

# County_Health = pd.read_csv('County_Health.csv', encoding='latin1')
inputdir2 = f"{homedir}" + "/data/us/other/"
Policies = pd.read_csv(inputdir2 + 'policies.csv', encoding='latin1')
Transit = pd.read_csv(inputdir2 + 'transit.csv', encoding='latin1')

# Policy_Transit = pd.read_csv('Policy_Transit.csv', encoding='latin1')


#Merging of Mainly Fixed Data

#Merging Policy & Educ, have most rows
Policy_Educ = pd.merge(Educ, Policies, left_on='FIPS', right_on='FIPS')
#Merging with Area Data, also has some extra rows
Policy_Educ_Area = pd.merge(Policy_Educ, Area_Houses, left_on='FIPS', right_on='County FIPS')
# #Merging with Health Data
# Policy_Educ_Area_Health = pd.merge(Policy_Educ_Area, County_Health, left_on='FIPS', right_on='FIPS')
#Merging with Population Data
Policy_Educ_Area_Health_Pop = pd.merge(Policy_Educ_Area_Health, County_Pop_Pop60, left_on='FIPS', right_on='FIPS')
#Merging with Unemployment Data
Policy_Educ_Area_Health_Pop_Unemp = pd.merge(Policy_Educ_Area_Health_Pop, Unemp, left_on='FIPS', right_on='FIPS')
#Merging with Poverty Data
Policy_Educ_Area_Health_Pop_Unemp_Pov = pd.merge(Policy_Educ_Area_Health_Pop_Unemp, Poverty, left_on='FIPS', right_on='FIPS')
#Merging with Transit Data
Policy_Educ_Area_Health_Pop_Unemp_Pov_Transit = pd.merge(Policy_Educ_Area_Health_Pop_Unemp_Pov, Transit, left_on='FIPS', right_on='FIPS')
#Merging with Votes Data
Policy_Educ_Area_Health_Pop_Unemp_Pov_Transit_Votes = pd.merge(Policy_Educ_Area_Health_Pop_Unemp_Pov_Transit, Votes, left_on='FIPS', right_on='FIPS')

#Same as above DataFrame, dataset of merged Policy, Education, Area, Health, Population, Poverty, Transit, & Voting Data
#Organized by FIPS (Nearly all FIPS codes accounted for)
Merged_Fixed = Policy_Educ_Area_Health_Pop_Unemp_Pov_Transit_Votes
Merged_Fixed.to_csv(outputdir+'Merged_Fixed.csv')

#Joining Merged Data with Racial/Age Data (This is incomplete, not all FIPS included)
Merged_Demographics = pd.merge(Merged_Fixed, Age_Race, left_on='FIPS', right_on='FIPS')
Merged_Demographics.to_csv(outputdir+'Merged_Demographics.csv')

#Merging of Fixed Data with Death Data

#Merging CD with Policy, Educ, have extra rows
Policy_Educ_CD = pd.merge(Policy_Educ, USAFacts_CDConsecutive, left_on='FIPS', right_on='countyFIPS')
#Merging with Area Data, also has some extra rows
Policy_Educ_Area_CD = pd.merge(Policy_Educ_CD, Area_Houses, left_on='FIPS', right_on='County FIPS')

# # #Merging with Health Data
# # Policy_Educ_Area_Health_CD = pd.merge(Policy_Educ_Area_CD, County_Health, left_on='FIPS', right_on='FIPS')
# #Merging with Population Data
# Policy_Educ_Area_Health_Pop_CD = pd.merge(Policy_Educ_Area_Health_CD, County_Pop_Pop60, left_on='FIPS', right_on='FIPS')
# #Merging with Unemployment Data
# Policy_Educ_Area_Health_Pop_Unemp_CD = pd.merge(Policy_Educ_Area_Health_Pop_CD, Unemp, left_on='FIPS', right_on='FIPS')
# #Merging with Poverty Data
# Policy_Educ_Area_Health_Pop_Unemp_Pov_CD = pd.merge(Policy_Educ_Area_Health_Pop_Unemp_CD, Poverty, left_on='FIPS', right_on='FIPS')
# #Merging with Transit Data
# Policy_Educ_Area_Health_Pop_Unemp_Pov_Transit_CD = pd.merge(Policy_Educ_Area_Health_Pop_Unemp_Pov_CD, Transit, left_on='FIPS', right_on='FIPS')
# #Merging with Votes Data
# Policy_Educ_Area_Health_Pop_Unemp_Pov_Transit_Votes_CD = pd.merge(Policy_Educ_Area_Health_Pop_Unemp_Pov_Transit_CD, Votes, left_on='FIPS', right_on='FIPS')


# USAFacts_CDConsecutive = pd.read_csv('USAFacts_CDConsecutive.csv', encoding='latin1')
# USAFacts_CDNonConsecutive = pd.read_csv('USAFacts_CDNonConsecutive.csv', encoding='latin1')
# USAFacts_CDConsecutive = USAFacts_CDConsecutive.drop(columns=['Unnamed: 0', 'County Name_C', 'State_C'])
# USAFacts_CDNonConsecutive = USAFacts_CDNonConsecutive.drop(columns=['Unnamed: 0', 'County Name_C', 'State_C'])

# #Making Merged Dataset
# Merged_CD = pd.merge(Merged_Fixed, USAFacts_CDConsecutive, left_on='FIPS', right_on='countyFIPS')
# Merged_CD.to_csv(outputdir+'Merged_CD.csv')

# #Joining Merged Data with Racial/Age Data (This is incomplete, not all FIPS included)
# Merged_Demographics_CD = pd.merge(Merged_CD, Age_Race, left_on='FIPS', right_on='FIPS')
# Merged_Demographics_CD.to_csv(outputdir+'Merged_Demographics_CD.csv')



