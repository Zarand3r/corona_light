import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import git

repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
datadir = f"{homedir}/data/international/italy/covid/"

# translate regional file
dfr = pd.read_csv(datadir + "dpc-covid19-ita-regioni.csv")
dfr.columns = ["Date","Country", "Regional Code", "Region", "Latitude","Longitude","HospitalizedWithSymptoms","IntensiveCare","TotalHospitalized","HomeIsolation","TotalCurrentlyPositive","ChangeTotalPositive","NewCurrentlyPositive","DischargedHealed","Deaths","TotalCases","Tested","Note_IT","Note_ENG"]
# dfr.to_csv(datadir + 'dpc-covid19-ita-regioni.csv', index=False)

# translate provincial file
dfp = pd.read_csv(datadir + "dpc-covid19-ita-province.csv")
dfp.columns = ["Date","Country", "Regional Code", "Region", "Province Code","Province","ProvinceInitials","Latitude","Longitude","TotalCases","Note_IT","Note_ENG"]
# dfp.to_csv(datadir + "dpc-covid19-ita-province.csv", index=False)

def load_region(region):
	region_data = dfr[dfr["Region"]==region]
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
	abruzzo = load_region("Abruzzo")
	plot_features(abruzzo, "TotalHospitalized","HomeIsolation","TotalCurrentlyPositive","Deaths","TotalCases")
