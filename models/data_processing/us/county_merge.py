import pandas as pd
import numpy as np
import os

def main():
	Key =  pd.read_csv('county_key.csv', index_col=0).sort_values(by=['FIPS'])

	#County Covid Cases/Deaths
	#NaN cleaning here set things to 0
	Consecutive_CD =  pd.read_csv('USAFacts_CDConsecutive.csv', index_col=0).sort_values(by=['FIPS'])
	Nonconsecutive_CD =  pd.read_csv('USAFacts_CDNonconsecutive.csv', index_col=0).sort_values(by=['FIPS'])

	#County Health Info
	County_Beds =  pd.read_csv('County_Beds.csv', index_col=0).sort_values(by=['FIPS'])
	County_ICU =  pd.read_csv('County_ICU.csv', index_col=0).sort_values(by=['FIPS'])
	County_Health =  pd.read_csv('County_Health.csv', index_col=0).sort_values(by=['FIPS'])

	#County Mobility Info (Need to be cleaned for NaNs)
	Consecutive_M =  pd.read_csv('Mobility_County_Consecutive.csv', index_col=0).sort_values(by=['FIPS'])
	Nonconsecutive_M =  pd.read_csv('Mobility_County_Nonconsecutive.csv', index_col=0).sort_values(by=['FIPS'])
	#making NaN=0
	Consecutive_M = Consecutive_M.fillna(0)
	Nonconsecutive_M = Nonconsecutive_M.fillna(0)
	#######################################################

	#Google Mobility Info
	#This needs to be cleaned for NaNs
	google_county_Consecutive = pd.read_csv('google_county_Consecutive.csv', index_col=0).sort_values(by=['FIPS'])
	google_county_Nonconsecutive = pd.read_csv('google_county_Nonconsecutive.csv', index_col=0).sort_values(by=['FIPS'])

	#This needs to be cleaned for NaNs
	Policies_County = pd.read_csv('Policies_County.csv', index_col=0).sort_values(by=['FIPS']) 

	Transit = pd.read_csv('Transit.csv', index_col=0).sort_values(by=['FIPS'])
	#County Demographic Info
	Votes =  pd.read_csv('Votes.csv', index_col=0).sort_values(by=['FIPS'])
	Age_Race =  pd.read_csv('Age_Race.csv', index_col=0).sort_values(by=['FIPS'])
	Educ_County =  pd.read_csv('Educ_County.csv', index_col=0).sort_values(by=['FIPS'])
	Density =  pd.read_csv('Density.csv', index_col=0).sort_values(by=['FIPS'])
	Unemp =  pd.read_csv('Unemp.csv', index_col=0).sort_values(by=['FIPS'])
	Poverty =  pd.read_csv('Poverty.csv', index_col=0).sort_values(by=['FIPS'])
	Pop_60 =  pd.read_csv('Pop_60.csv', index_col=0).sort_values(by=['FIPS'])

	#County Air Quality Info
	#This needs to be cleaned for NaNs
	Air_Qual = pd.read_csv('Air_Qual.csv', index_col=0).sort_values(by=['FIPS','ValidDate'])
	Air_Qual = Air_Qual.set_index('FIPS')

	#Individual breakdown of Air Quality, needs to be cleaned for NaNs
	Ozone_AQI = pd.read_csv('Ozone_AQI.csv', index_col=0).sort_values(by=['FIPS'])
	PM10_AQI = pd.read_csv('PM10_AQI.csv', index_col=0).sort_values(by=['FIPS'])
	PM25_AQI = pd.read_csv('PM25_AQI.csv', index_col=0).sort_values(by=['FIPS'])
	NO2_AQI = pd.read_csv('NO2_AQI.csv', index_col=0).sort_values(by=['FIPS'])
	CO_PPB = pd.read_csv('CO_PPB.csv', index_col=0).sort_values(by=['FIPS'])
	SO2_PPB = pd.read_csv('SO2_PPB.csv', index_col=0).sort_values(by=['FIPS'])


if __name__ == '__main__':
	main()
