import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import git


def load_data(filename, directory="/data/international/italy/covid/"):
	repo = git.Repo("./", search_parent_directories=True)
	homedir = repo.working_dir
	filepath = f"{homedir}" + directory + filename
	df = pd.read_csv(filepath)
	return df 

def query(data, query_key, query_val):
	query_data = data[data[query_key]==query_val]
	return query_data

def plot_features(dataframe, *features):
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
	italy = load_data("dpc-covid19-ita-regioni.csv", "/models/processing/International/Italy/")
	abruzzo = query(italy, "Region", "Abruzzo")
	plot_features(abruzzo, "TotalHospitalized","HomeIsolation","TotalCurrentlyPositive","Deaths","TotalCases")
