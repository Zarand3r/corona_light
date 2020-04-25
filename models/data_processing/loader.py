import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import git


def load_data(filename, directory="", encoding=None):
	repo = git.Repo("./", search_parent_directories=True)
	homedir = repo.working_dir
	if len(directory) == 0:
		filepath = f"{homedir}" + filename
	else:
		filepath = f"{homedir}" + directory + filename
	dataframe = pd.read_csv(filepath, encoding=encoding)
	return dataframe 

# TODO make this more dynamic
# Detect in load_data if a dataframe has a key with timestamp values, and if so call this function
def convert_dates(dataframe, timestampkey):
	dataframe['date_processed'] = pd.to_datetime(dataframe[timestampkey].values)
	dataframe['date_processed'] = (dataframe['date_processed'] - dataframe['date_processed'].min())/np.timedelta64(1, 'D')

def query(dataframe, query_key, query_val, reset=True):
	query_data = dataframe[dataframe[query_key]==query_val]
	if len(query_data) == 0:
		print("this is empty")
	if reset:
		query_data.reset_index(drop=True, inplace=True)
	return query_data

def plot_features(dataframe, *features):
	# print(dataframe["date_processed"])
	time_list = []
	datetimeFormat = '%Y-%m-%dT%H:%M:%S'
	initial_time = dataframe["Date"][0]
	for index, row in dataframe.iterrows():
		current_time = row["Date"]
		delta = datetime.datetime.strptime(current_time, datetimeFormat)-datetime.datetime.strptime(initial_time, datetimeFormat)
		delta = delta.days
		time_list.append(delta)

	for feature in features:
		feature_data = dataframe[feature]
		plt.plot(time_list, feature_data)

	plt.show()

if __name__ == '__main__':
	italy = load_data("dpc-covid19-ita-regioni.csv", "/models/data/international/italy/covid/")
	# italy = load_data("/models/data/international/italy/covid/dpc-covid19-ita-regioni.csv")
	lombardia = query(italy, "Region", "Lombardia")
	plot_features(lombardia, "TotalHospitalized","HomeIsolation","TotalCurrentlyPositive","Deaths","TotalCases")



# #example usage from outside file:
# import sys
# sys.path.insert(1, '../processing')
# import loader

# italy = loader.load_data("dpc-covid19-ita-regioni.csv", "/models/processing/International/Italy/")
# abruzzo = loader.query(italy, "Region", "Abruzzo")
# loader.plot_features(abruzzo, "TotalHospitalized","HomeIsolation","TotalCurrentlyPositive","Deaths","TotalCases")