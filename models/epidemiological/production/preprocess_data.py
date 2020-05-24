import itertools
import os

import math
import pandas as pd
import numpy as np
import datetime

import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/models/data_processing')
import loader

def add_active_cases(us, data_active_cases):
	active_cases = loader.load_data(data_active_cases)
	active_cases['FIPS']=active_cases['FIPS'].astype(int)
	loader.convert_dates(active_cases, "Date")
	difference = (pd.to_datetime(active_cases['Date'])[0] - pd.to_datetime(us['date'])[0])/np.timedelta64(1, 'D')

	active_column = []
	end = len(us)-1
	for index, row in us.iterrows():
		print(f"{index}/{end}")
		county = row['fips']
		date = row['date_processed'] 
		if date < difference:
			active_column.append(-1)
		else:
			entry = (active_cases[(active_cases.date_processed==date-difference) & (active_cases.FIPS == county)])["Active"].values
			if len(entry) != 0:
				active_column.append(entry[0])
			else:
				active_column.append(-1)

	us["active_cases"] = active_column
	return us


def process_data(data_covid, data_population, save=True):
	covid = loader.load_data(data_covid)
	loader.convert_dates(covid, "date")
	population = loader.load_data(data_population)
	covid.loc[covid["county"]=='New York City', "fips"]=36061
	covid['Population'] = covid.apply(lambda row: loader.query(population, "FIPS", row.fips)['total_pop'], axis=1)
	covid.dropna(subset=['fips'], inplace=True)
	covid['fips']=covid['fips'].astype(int)
	covid = add_active_cases(covid, "/data/us/covid/JHU_daily_US.csv")
	if save:
		covid.to_csv(f"{homedir}" + "/models/epidemiological/production/us_training_data.csv")
	return covid


if __name__ == '__main__':
	us = process_data("/data/us/covid/nyt_us_counties.csv", "/data/us/demographics/county_populations.csv", save=True)


