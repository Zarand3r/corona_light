import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import csv


import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
# FIXES NANs

def fix_nans(submission, save=False):
	submission.fillna(method='ffill', inplace=True)
	num = submission._get_numeric_data()
	num[num < 0.1] = 0
	if save:
		submission.to_csv("modified_submission.csv", index=False)

def reformat(file1, file2, save=True):
	submission = pd.read_csv(file1, index_col=False)
	modified_submission = pd.read_csv(file2, index_col=False)
	fix_nans(submission)
	for index, row in modified_submission.iterrows():
		current_id = row["id"]
		replacement = [current_id, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		forecast = submission[submission["id"]==current_id].values
		if len(forecast) > 0:
			replacement = forecast[0]
		modified_submission.loc[index] = replacement
		# modified_submission.loc[index, "id"] = (submission[submission["id"]==current_id]).values[0]
	if save:
		modified_submission.to_csv("modified_submission.csv", index=False)

def reformat2(file1, file2, save=True):
	submission = pd.read_csv(file1, index_col=False)
	modified_submission = pd.read_csv(file2, index_col=False)
	fix_nans(submission)

	final_submission = []

	forecast_dict = {}
	for index, row in submission.iterrows():
		current_id = row["id"]
		forecast_dict[current_id] = submission[submission["id"]==current_id].values[0]

	for index, row in modified_submission.iterrows():
		current_id = row["id"]
		replacement = forecast_dict.pop(current_id, [current_id, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 
		final_submission.append(replacement)

	if save:
		header = ["id", "10", "20", "30", "40", "50", "60", "70", "80", "90"]
		with open('modified_submission2.csv', 'w') as submission_file:
			writer = csv.writer(submission_file, delimiter=',')
			writer.writerow(header)
			writer.writerows(final_submission)

	

if __name__ == '__main__':

	reformat2("../submission1.csv", f"{homedir}/sample_submission.csv")
