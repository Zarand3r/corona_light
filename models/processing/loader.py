import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import git


def load_region(filename, query_key, query_val, directory="/data/international/italy/covid/"):
	repo = git.Repo("./", search_parent_directories=True)
	homedir = repo.working_dir
	filepath = f"{homedir}" + directory + filename
	print(filepath)
	df = pd.read_csv(filepath)
	region_data = df[df[query_key]==query_val]
	return region_data

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
	abruzzo = load_region("dpc-covid19-ita-regioni.csv", "Region", "Abruzzo", "/models/processing/International/Italy/")
	plot_features(abruzzo, "TotalHospitalized","HomeIsolation","TotalCurrentlyPositive","Deaths","TotalCases")
